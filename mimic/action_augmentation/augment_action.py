"""Action augmentation: video + label in, deviated video + corrective label out.

Builds a deviate-and-recover maneuver on a recorded clip — the robot drifts
laterally to a peak offset by ``horizon / 2`` and rejoins the recorded path by
``horizon`` — then re-renders the clip from that camera path with
`TrajectoryCrafter <https://github.com/TrajectoryCrafter/TrajectoryCrafter>`_
(submodule under ``third_party/``, used unmodified) and writes the matching
action label.

Only the lateral offset is specified; heading follows from the path tangent, so
the maneuver stays kinematically consistent. See :mod:`.trajectory` for the math.

Usage:
    # 0.5 m drift left, rejoining over a 4 s horizon
    python -m mimic.action_augmentation.augment_action \\
        --input clip.mp4 --strength 0.5

    # sample the offset per clip from +/-[0.3, 0.8] m
    python -m mimic.action_augmentation.augment_action \\
        --input clip.mp4 --strength_range 0.3 0.8 --seed 42

    # trajectory and label only, no video synthesis (no GPU needed)
    python -m mimic.action_augmentation.augment_action \\
        --input clip.mp4 --strength 0.5 --labels_only

Video synthesis needs a CUDA GPU and the TrajectoryCrafter runtime
(``uv sync --extra action``). The label path is pure numpy.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import labels as label_io
from . import trajectory as tj

#: This package.
PACKAGE_DIR = Path(__file__).resolve().parent

#: TrajectoryCrafter submodule (upstream code, unmodified).
DEFAULT_TC_ROOT = PACKAGE_DIR / "third_party" / "TrajectoryCrafter"

#: Weight cache. TrajectoryCrafter pulls its checkpoints from HuggingFace.
DEFAULT_MODELS_DIR = PACKAGE_DIR / "models"

#: TrajectoryCrafter renders a fixed-length window.
TC_VIDEO_LENGTH = 49


def resolve_tc_root(explicit: str | None = None) -> Path:
    """Locate the TrajectoryCrafter checkout: ``--tc_root``, ``$TC_ROOT``, then the submodule."""
    for candidate in (explicit, os.environ.get("TC_ROOT"), DEFAULT_TC_ROOT):
        if not candidate:
            continue
        root = Path(candidate).resolve()
        if (root / "demo.py").is_file():
            return root
    raise FileNotFoundError(
        f"TrajectoryCrafter not found at {DEFAULT_TC_ROOT}. The submodule is probably "
        "not initialized — run:\n"
        "  git submodule update --init --recursive\n"
        "Or point at an existing checkout with --tc_root / $TC_ROOT."
    )


def clip_seed(base_seed: int, video_path: str) -> int:
    """Deterministic per-clip seed, so a batch under one seed still varies."""
    digest = int(hashlib.sha256(str(video_path).encode()).hexdigest(), 16)
    return (base_seed ^ digest) & 0xFFFF_FFFF


def sample_strength(
    seed: int,
    strength: float | None = None,
    strength_range: tuple[float, float] | None = None,
) -> float:
    """Pick the peak lateral offset in meters. Sign (left/right) is coin-flipped."""
    if strength is not None:
        return float(strength)
    if strength_range is None:
        raise ValueError("pass either strength or strength_range")
    lo, hi = sorted(abs(v) for v in strength_range)
    rng = random.Random(seed)
    return rng.uniform(lo, hi) * rng.choice((-1.0, 1.0))


# =====================================================================
# Camera poses for TrajectoryCrafter
# =====================================================================


def camera_deltas(
    original: np.ndarray,
    augmented: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame camera delta of the augmented path relative to the recorded one.

    The recorded clip already contains the robot's forward motion, so only the
    *difference* is imposed on the renderer.

    Returns:
        ``(lateral, yaw)`` — lateral offset in meters (positive left) and yaw
        delta in radians (positive counter-clockwise), both ``(N,)``.
    """
    original = np.asarray(original, dtype=np.float64)
    augmented = np.asarray(augmented, dtype=np.float64)
    if original.shape != augmented.shape:
        raise ValueError(f"shape mismatch: {original.shape} vs {augmented.shape}")

    lateral = np.empty(len(original))
    for i in range(len(original)):
        lateral[i] = tj.relative_pose(original[i], augmented[i])[1]
    yaw = tj.wrap_angle(augmented[:, 2] - original[:, 2])
    return lateral, yaw


def build_c2w_poses(
    tc_module,
    lateral: np.ndarray,
    yaw: np.ndarray,
    c2w_init,
    scale: float,
    device: str,
):
    """Turn metric lateral/yaw deltas into TrajectoryCrafter camera-to-world poses.

    Upstream's ``sphere2pose`` takes a translation along camera ``x`` and a
    rotation ``phi`` about camera ``y``. Ours maps onto both directly: lateral
    offset is the ``x`` translation, heading change is ``phi``.

    ``scale`` converts meters into the reconstruction's depth units, which are
    not metric — see ``--scale`` in the CLI.
    """
    import torch

    poses = []
    for lat, psi in zip(lateral, yaw):
        poses.append(
            tc_module.sphere2pose(
                c2w_init,
                theta=np.float32(0.0),
                # sphere2pose applies phi about the camera's up axis in degrees.
                phi=np.float32(np.degrees(psi)),
                r=np.float32(0.0),
                device=device,
                # x is negated inside sphere2pose, so +x here is a left shift.
                x=np.float32(lat * scale),
                y=np.float32(0.0),
            )
        )
    return torch.cat(poses, dim=0)


def make_get_poses(lateral: np.ndarray, yaw: np.ndarray, scale: float):
    """Build a ``get_poses`` replacement that injects our camera path.

    Upstream's own ``get_poses`` only supports a single interpolated target pose
    or a spherical theta/phi/r keyframe file; neither expresses a per-frame
    ground-plane maneuver. Binding this in place of it leaves the submodule
    untouched.
    """

    def get_poses(self, opts, depths, num_frames):
        import torch
        from models.utils import sphere2pose  # from the TrajectoryCrafter submodule

        tc_module = SimpleNamespace(sphere2pose=sphere2pose)

        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
        cx, cy, f = 512.0, 288.0, 500.0
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(opts.device)
        )
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(opts.device)
            .unsqueeze(0)
        )

        # Resample our per-frame deltas onto the renderer's frame count.
        idx = np.linspace(0, len(lateral) - 1, num_frames)
        lat_r = np.interp(idx, np.arange(len(lateral)), lateral)
        yaw_r = np.interp(idx, np.arange(len(yaw)), np.unwrap(yaw))

        poses = build_c2w_poses(tc_module, lat_r, yaw_r, c2w_init, scale, opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        return pose_s, poses, K

    return get_poses


# =====================================================================
# CLI
# =====================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Action augmentation: deviate-and-recover maneuvers with matching labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--input", type=str, required=True, help="Input video (.mp4).")
    io_group.add_argument("--labels", type=str, default=None,
                          help="Label sidecar. Default: <stem>.npy/.npz/.json beside the video.")
    io_group.add_argument("--output", type=str, default=None,
                          help="Output video. Default: <stem>_act.mp4 beside the input.")
    io_group.add_argument("--output_labels", type=str, default=None,
                          help="Output sidecar. Default: matches --output's stem, .npz.")
    io_group.add_argument("--tc_root", type=str, default=None,
                          help="TrajectoryCrafter checkout. Default: the third_party submodule.")
    io_group.add_argument("--fps", type=float, default=label_io.DEFAULT_FPS,
                          help="Frame rate assumed when the sidecar has no timestamps.")
    io_group.add_argument("--overwrite", action="store_true")
    io_group.add_argument("--labels_only", action="store_true",
                          help="Compute the trajectory and label, skip video synthesis (no GPU).")
    io_group.add_argument("--dry_run", action="store_true",
                          help="Print the plan and exit without writing anything.")

    man = parser.add_argument_group("maneuver")
    man.add_argument("--strength", type=float, default=None,
                     help="Peak lateral offset in meters. Positive left, negative right.")
    man.add_argument("--strength_range", type=float, nargs=2, default=None,
                     metavar=("LO", "HI"),
                     help="Sample |offset| from [LO, HI] and coin-flip the side.")
    man.add_argument("--horizon", type=float, default=tj.DEFAULT_HORIZON_S,
                     help="Maneuver duration in seconds; the peak lands at half of it.")
    man.add_argument("--start_time", type=float, default=0.0,
                     help="When the maneuver begins, in clip time.")
    man.add_argument("--profile", type=str, default="raised_cosine", choices=tj.PROFILES,
                     help="Lateral offset profile.")
    man.add_argument("--mode", type=str, default="deviate_recover",
                     choices=["deviate_recover", "reexpress"],
                     help="deviate_recover builds the S-curve maneuver; reexpress holds a "
                          "constant offset and labels the way back to the recorded path.")
    man.add_argument("--seed", type=int, default=42)
    man.add_argument("--fixed_seed", action="store_true",
                     help="Use --seed directly instead of mixing in the input path.")

    render = parser.add_argument_group("rendering")
    render.add_argument("--scale", type=float, default=1.0,
                        help="Depth units per meter. TrajectoryCrafter's depth is not metric, "
                             "so a lateral offset in meters must be converted. Do not guess it: "
                             "run `python -m mimic.action_augmentation.calibrate_scale` to read "
                             "it off the action labels, then keep it fixed.")
    render.add_argument("--radius_scale", type=float, default=1.0)
    render.add_argument("--stride", type=int, default=1)
    render.add_argument("--diffusion_inference_steps", type=int, default=50)
    render.add_argument("--diffusion_guidance_scale", type=float, default=6.0)
    render.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args(argv)
    if args.strength is None and args.strength_range is None:
        parser.error("pass --strength or --strength_range")
    if args.strength is not None and args.strength_range is not None:
        parser.error("--strength and --strength_range are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"error: input video not found: {input_path}", file=sys.stderr)
        return 1

    try:
        sidecar = label_io.find_sidecar(input_path, Path(args.labels) if args.labels else None)
        data = label_io.load_labels(sidecar, fps=args.fps)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_act.mp4")
    )
    if output_path == input_path:
        print("error: --output would overwrite the input video", file=sys.stderr)
        return 1
    output_labels = (
        Path(args.output_labels).expanduser().resolve()
        if args.output_labels
        else output_path.with_suffix(".npz")
    )
    if output_labels.resolve() == sidecar.resolve():
        print("error: --output_labels would overwrite the input sidecar", file=sys.stderr)
        return 1
    for existing in (output_path, output_labels):
        if existing.exists() and not args.overwrite and not args.dry_run:
            if existing == output_path and args.labels_only:
                continue
            print(f"error: {existing} exists (pass --overwrite)", file=sys.stderr)
            return 1

    seed = args.seed if args.fixed_seed else clip_seed(args.seed, str(input_path))
    strength = sample_strength(
        seed, args.strength, tuple(args.strength_range) if args.strength_range else None
    )

    result = tj.build_augmented_label(
        data.poses, data.times,
        strength=strength, horizon=args.horizon, start_time=args.start_time,
        profile=args.profile, label_times=data.label_times, mode=args.mode,
    )
    lateral, yaw = camera_deltas(data.poses, result["poses"])

    print(f"Input:    {input_path}")
    print(f"Labels:   {sidecar.name}"
          + ("  (poses reconstructed from waypoints — approximate)" if data.reconstructed else ""))
    print(f"Clip:     {len(data)} frames, {data.duration:.2f}s")
    print(f"Maneuver: {args.mode}, {strength:+.2f} m peak "
          f"({'left' if strength >= 0 else 'right'}), {args.horizon:.1f}s horizon, {args.profile}")
    print(f"Seed:     {seed}" + ("" if args.fixed_seed else f" (derived from {args.seed} + path)"))
    print(f"Yaw:      {np.degrees(np.abs(yaw)).max():.2f} deg peak")
    print(f"Label:    {result['waypoints'].shape} waypoints -> {output_labels.name}")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    written = label_io.save_labels(
        output_labels, result["poses"], data.times,
        result["waypoints"], result["label_times"],
        metadata={
            "source_video": str(input_path),
            "source_labels": str(sidecar),
            "mode": args.mode,
            "strength_m": strength,
            "horizon_s": args.horizon,
            "start_time_s": args.start_time,
            "profile": args.profile,
            "seed": seed,
            "poses_reconstructed": data.reconstructed,
        },
    )
    print(f"Wrote {written}")

    if args.labels_only:
        print("Labels only: video synthesis skipped.")
        return 0

    try:
        tc_root = resolve_tc_root(args.tc_root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return _render(args, tc_root, input_path, output_path, lateral, yaw, seed)


def _render(args, tc_root: Path, input_path: Path, output_path: Path,
            lateral: np.ndarray, yaw: np.ndarray, seed: int) -> int:
    """Run TrajectoryCrafter with our camera path bound in place of its own."""
    import torch

    # Upstream resolves sibling modules relatively and reads/writes under cwd.
    if str(tc_root) not in sys.path:
        sys.path.insert(0, str(tc_root))
    cwd = Path.cwd()
    os.chdir(tc_root)
    try:
        from demo import TrajCrafter

        opts = SimpleNamespace(
            video_path=str(input_path),
            out_dir=str(output_path.parent),
            exp_name=output_path.stem,
            save_dir=str(output_path.parent / output_path.stem),
            device=args.device,
            seed=seed,
            video_length=TC_VIDEO_LENGTH,
            fps=round(args.fps),
            stride=args.stride,
            radius_scale=args.radius_scale,
            camera="traj",
            mode="gradual",
            mask=True,
            traj_txt=None,
            target_pose=None,
            near=0.0001,
            far=10000.0,
            anchor_idx=0,
            low_gpu_memory_mode=False,
            model_name="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
            transformer_path="TrajectoryCrafter/TrajectoryCrafter",
            sampler_name="DDIM_Origin",
            sample_size=[384, 672],
            diffusion_guidance_scale=args.diffusion_guidance_scale,
            diffusion_inference_steps=args.diffusion_inference_steps,
            prompt=None,
            negative_prompt=(
                "The video is not of a high quality, it has a low resolution. "
                "Watermark present in each frame. The background is solid. "
                "Strange body and strange trajectory. Distortion."
            ),
            refine_prompt=(
                ". The video is of high quality, and the view is very clear. "
                "High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
            ),
            blip_path="Salesforce/blip2-opt-2.7b",
            unet_path="tencent/DepthCrafter",
            pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
            cpu_offload="model",
            depth_inference_steps=5,
            depth_guidance_scale=1.0,
            window_size=110,
            overlap=25,
            max_res=1024,
            weight_dtype=torch.bfloat16,
        )
        os.makedirs(opts.save_dir, exist_ok=True)

        crafter = TrajCrafter(opts)
        # Bind our camera path in place of upstream's pose generator.
        crafter.get_poses = make_get_poses(lateral, yaw, args.scale).__get__(crafter)
        crafter.infer_gradual(opts)
    finally:
        os.chdir(cwd)

    print(f"Rendered into {output_path.parent / output_path.stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
