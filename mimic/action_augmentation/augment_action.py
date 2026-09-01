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
from . import scales as scale_io
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
    strength_scale: float | None = None,
    speed: float | None = None,
) -> float:
    """Pick the peak lateral offset in meters. Sign (left/right) is coin-flipped.

    Exactly one of the three sizing options applies, in this order:

    ``strength``
        An absolute offset in meters, used as given.
    ``strength_scale``
        An offset relative to ``speed``, as ``scale * speed`` — see
        :func:`~mimic.action_augmentation.trajectory.strength_for_speed`. Only
        the side is drawn; the magnitude follows the clip.
    ``strength_range``
        An absolute offset drawn from ``[lo, hi]`` meters.
    """
    if strength is not None:
        return float(strength)

    rng = random.Random(seed)
    if strength_scale is not None:
        if speed is None:
            raise ValueError("strength_scale needs the clip's speed")
        return tj.strength_for_speed(strength_scale, speed) * rng.choice((-1.0, 1.0))

    if strength_range is None:
        raise ValueError("pass strength, strength_scale or strength_range")
    lo, hi = sorted(abs(v) for v in strength_range)
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


def _patch_write_video_for_torchvision() -> None:
    """Restore ``torchvision.io.write_video``, removed in torchvision 0.22.

    Upstream saves every result with it. Newer torchvision dropped the whole
    video-writing API, so the pipeline runs to completion and then fails on the
    last line with an AttributeError.

    The replacement writes through imageio's ffmpeg backend with upstream's own
    codec and CRF, so the output file is what upstream intended.
    """
    import torchvision.io

    if hasattr(torchvision.io, "write_video"):
        return

    def write_video(filename, video_array, fps, video_codec="libx264", options=None):
        import imageio.v2 as imageio
        import numpy as np

        frames = video_array
        if hasattr(frames, "cpu"):
            frames = frames.cpu().numpy()
        frames = np.asarray(frames, dtype=np.uint8)

        params = ["-crf", str((options or {}).get("crf", "10"))]
        codec = "libx264" if video_codec in ("h264", "libx264") else video_codec
        writer = imageio.get_writer(
            str(filename), fps=fps, codec=codec, quality=None,
            macro_block_size=1, pixelformat="yuv420p", ffmpeg_params=params,
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()

    torchvision.io.write_video = write_video


def _patch_pipeline_device_for_offload(crafter) -> None:
    """Make ``pipeline.device`` usable while the pipeline is offloaded.

    ``enable_sequential_cpu_offload`` leaves every parameter on the meta device
    until its block is needed, so ``DiffusionPipeline.device`` — which reports
    the device of the first module it finds — answers ``meta``. Upstream builds
    its cross-attention reference with ``ref_latents.to(device=self.device)``,
    which quietly moves real data onto meta and throws it away; the failure only
    surfaces later as "Cannot copy out of meta tensor" inside accelerate.

    Sequential offload is not optional here: the refinement transformer is ~10 GB
    in bf16, so on a 16 GB card whole-model offload cannot fit the activations.

    Falls back to the original property whenever it reports a real device, so
    nothing changes for the un-offloaded path.
    """
    pipeline = getattr(crafter, "pipeline", None)
    if pipeline is None:
        return
    cls = type(pipeline)
    if getattr(cls, "_mimic_device_patched", False):
        return

    original = cls.device

    def device(self):
        resolved = original.fget(self)
        if getattr(resolved, "type", None) == "meta":
            return self._execution_device
        return resolved

    cls.device = property(device)
    cls._mimic_device_patched = True


def _enable_vae_memory_savers(crafter) -> None:
    """Turn on the pipeline's own VAE slicing and tiling if it has them.

    Both trade a little speed for a much smaller peak during encode and decode,
    and neither changes the result. Decoding 49 frames in one piece asks for a
    single ~4.7 GB block, which is the difference between finishing and not on a
    16 GB card.

    Upstream's pipeline predates the ``enable_vae_*`` convenience wrappers, so
    the VAE's own methods are the ones that matter here; the wrappers are tried
    first for pipelines that do have them.
    """
    pipeline = getattr(crafter, "pipeline", None)
    targets = [
        (pipeline, "enable_vae_slicing"),
        (pipeline, "enable_vae_tiling"),
        (getattr(pipeline, "vae", None), "enable_slicing"),
        (getattr(pipeline, "vae", None), "enable_tiling"),
    ]
    for owner, method in targets:
        function = getattr(owner, method, None)
        if callable(function):
            try:
                function()
            except Exception as exc:  # noqa: BLE001 - a saver is optional
                print(f"note: {method} unavailable ({exc})", flush=True)


def _release_captioner(crafter, prompt: str, refine_prompt: str) -> None:
    """Use a fixed prompt and give the captioner's memory back.

    Upstream builds BLIP-2 OPT-2.7b in ``__init__`` and pins it to the GPU with
    a plain ``.to(device)`` — no offload — purely to describe one frame. That is
    roughly 7.5 GB resident before DepthCrafter and the refinement model load,
    which is the difference between fitting and not fitting on a 16 GB card.

    When the caption is supplied there is nothing for it to do, so replace the
    method with the constant and drop the weights.

    The attributes are set to ``None`` rather than deleted: upstream frees them
    itself part-way through ``infer_gradual`` with ``del self.captioner``, which
    would raise if the name were already gone.
    """
    import gc

    import torch

    crafter.get_caption = lambda opts, image: prompt + refine_prompt
    crafter.captioner = None
    crafter.caption_processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _patch_sincos_for_diffusers() -> None:
    """Let the pinned submodule keep working on diffusers >= 0.33.

    Upstream calls ``get_3d_sincos_pos_embed`` without ``output_type``, which
    still defaults to ``"np"``. From 0.33 that path is deprecated, and by 0.40
    diffusers raises instead of warning, so the transformer cannot be built at
    all.

    The two branches are the same computation — identical construction, both
    returning ``[T, H*W, D]`` — so routing to the tensor branch and handing back
    an array is exact, not an approximation. Patching the name in the submodule's
    own namespace leaves the submodule's source untouched.
    """
    try:
        from diffusers.models import embeddings as diffusers_embeddings
    except ImportError:
        return

    try:
        import models.crosstransformer3d as upstream
    except ImportError:
        return

    if getattr(upstream, "_mimic_sincos_patched", False):
        return

    original = diffusers_embeddings.get_3d_sincos_pos_embed

    def get_3d_sincos_pos_embed(*args, **kwargs):
        if kwargs.get("output_type") is None:
            kwargs["output_type"] = "pt"
        result = original(*args, **kwargs)
        return result.cpu().numpy() if hasattr(result, "cpu") else result

    upstream.get_3d_sincos_pos_embed = get_3d_sincos_pos_embed
    upstream._mimic_sincos_patched = True


def make_get_poses(lateral: np.ndarray, yaw: np.ndarray, scale: float, stride: int = 1):
    """Build a ``get_poses`` replacement that injects our camera path.

    Upstream's own ``get_poses`` only supports a single interpolated target pose
    or a spherical theta/phi/r keyframe file; neither expresses a per-frame
    ground-plane maneuver. Binding this in place of it leaves the submodule
    untouched.

    ``stride`` must match what upstream's reader was given, because the deltas
    are indexed against the frames it actually read — the leading
    ``num_frames * stride`` of the clip — rather than stretched across the whole
    clip. Stretching them would hang a maneuver from one moment of the clip onto
    imagery from another.
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

        # Index the deltas against the frames upstream's reader actually took:
        # the leading num_frames * stride of the clip, sampled at stride.
        idx = np.clip(np.arange(num_frames) * stride, 0, len(lateral) - 1)
        lat_r = lateral[idx]
        yaw_r = np.unwrap(yaw)[idx]

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
    io_group.add_argument("--fps", type=float, default=None,
                          help="Frame rate for sidecars without timestamps. "
                               "Default: read from the clip's meta.json or container.")
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
    man.add_argument("--strength_scale", type=float, default=None,
                     help="Size the offset relative to the clip's speed, as "
                          "scale * max speed in the maneuver window (scale is in "
                          "seconds). Keeps the heading the maneuver demands constant "
                          f"across a corpus of mixed speeds; {tj.DEFAULT_STRENGTH_SCALE_S} "
                          "is about 15 deg over a 4 s horizon.")
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
    render.add_argument("--scale", type=str, default=scale_io.AUTO,
                        help="Depth units per meter, or 'auto' to read the clip's "
                             "<stem>.scale.json sidecar. TrajectoryCrafter's depth is not "
                             "metric, and its units depend on the clip, so a lateral offset "
                             "in meters must be converted per clip. Do not guess it: run "
                             "`python -m mimic.action_augmentation.calibrate_clips` to read "
                             "it off that clip's recorded poses.")
    render.add_argument("--scale_file", type=str, default=None,
                        help="Scale sidecar to use instead of the one beside the clip.")
    render.add_argument("--radius_scale", type=float, default=1.0)
    render.add_argument("--stride", type=int, default=1)
    render.add_argument("--diffusion_inference_steps", type=int, default=50)
    render.add_argument("--diffusion_guidance_scale", type=float, default=6.0)
    render.add_argument("--device", type=str, default="cuda:0")
    render.add_argument("--sample_size", type=int, nargs=2, default=(384, 672),
                        metavar=("H", "W"),
                        help="Refinement model's working size. Upstream's default is "
                             "384 672; the 5B transformer leaves little room for "
                             "activations on a 16 GB card, where 320 560 fits.")
    render.add_argument("--prompt", type=str, default=None,
                        help="Scene description for the refinement model. Default: "
                             "upstream captions the clip with BLIP-2, which pins "
                             "~7.5 GB to the GPU. Supplying one frees that, and is "
                             "effectively required below about 24 GB.")
    render.add_argument("--cpu_offload", type=str, default="model",
                        choices=("model", "sequential"),
                        help="DepthCrafter offload strategy.")
    render.add_argument("--low_gpu_memory_mode", action="store_true",
                        help="Upstream's low-VRAM path for the refinement "
                             "model. Needed below roughly 24 GB; slower.")

    args = parser.parse_args(argv)
    given = [name for name, value in (("--strength", args.strength),
                                      ("--strength_range", args.strength_range),
                                      ("--strength_scale", args.strength_scale))
             if value is not None]
    if not given:
        parser.error("pass --strength, --strength_range or --strength_scale")
    if len(given) > 1:
        parser.error(f"{' and '.join(given)} are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"error: input video not found: {input_path}", file=sys.stderr)
        return 1

    if args.fps is None:
        # A guessed rate rescales the whole timeline, so ask the clip first.
        args.fps = label_io.clip_fps(input_path) or label_io.DEFAULT_FPS

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
    window = (data.times >= args.start_time) & (
        data.times <= args.start_time + args.horizon
    )
    speeds = tj.path_speed(data.poses, data.times)
    reference_speed = float(speeds[window].max()) if window.any() else float(speeds.max())
    try:
        strength = sample_strength(
            seed, args.strength,
            tuple(args.strength_range) if args.strength_range else None,
            args.strength_scale, reference_speed,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

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
    sizing = (f"scale {args.strength_scale}s x {reference_speed:.2f} m/s"
              if args.strength_scale is not None else "absolute")
    print(f"Maneuver: {args.mode}, {strength:+.2f} m peak "
          f"({'left' if strength >= 0 else 'right'}), {args.horizon:.1f}s horizon, "
          f"{args.profile}, {sizing}")
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

    # Resolve before loading any model: rendering at an unverified scale
    # silently mis-sizes the maneuver, which is worse than not rendering.
    try:
        scale, provenance = scale_io.resolve_scale(
            input_path,
            args.scale,
            window=scale_io.make_window(TC_VIDEO_LENGTH, args.stride),
            path=args.scale_file,
        )
    except (scale_io.ScaleNotFound, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Scale:    {scale:.4f} depth units/m ({provenance})")

    covered = TC_VIDEO_LENGTH * args.stride
    active = np.flatnonzero(np.abs(lateral) > 1e-6)
    if len(active) and active[-1] >= covered:
        span = covered / max(args.fps, 1e-6)
        print(
            f"warning: the renderer reads only the leading {covered} frames "
            f"({span:.1f}s at {args.fps:g} fps), but the maneuver runs to frame "
            f"{active[-1]} (t={active[-1] / max(args.fps, 1e-6):.1f}s). The video will "
            f"not show it. Either move it with --start_time, or raise --stride to "
            f"{int(np.ceil((active[-1] + 1) / TC_VIDEO_LENGTH))} so the window reaches it.",
            file=sys.stderr,
        )

    return _render(args, tc_root, input_path, output_path, lateral, yaw, seed, scale)


def _render(args, tc_root: Path, input_path: Path, output_path: Path,
            lateral: np.ndarray, yaw: np.ndarray, seed: int, scale: float) -> int:
    """Run TrajectoryCrafter with our camera path bound in place of its own."""
    import gc

    import torch

    # Upstream resolves sibling modules relatively and reads/writes under cwd.
    if str(tc_root) not in sys.path:
        sys.path.insert(0, str(tc_root))
    cwd = Path.cwd()
    os.chdir(tc_root)
    crafter = None
    try:
        _patch_sincos_for_diffusers()
        _patch_write_video_for_torchvision()
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
            low_gpu_memory_mode=args.low_gpu_memory_mode,
            model_name="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
            transformer_path="TrajectoryCrafter/TrajectoryCrafter",
            sampler_name="DDIM_Origin",
            sample_size=list(args.sample_size),
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
            cpu_offload=args.cpu_offload,
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
        crafter.get_poses = make_get_poses(
            lateral, yaw, scale, args.stride
        ).__get__(crafter)
        if args.prompt:
            _release_captioner(crafter, args.prompt, opts.refine_prompt)
        _enable_vae_memory_savers(crafter)
        if args.low_gpu_memory_mode:
            _patch_pipeline_device_for_offload(crafter)
        crafter.infer_gradual(opts)
    finally:
        os.chdir(cwd)
        # A corpus run renders clip after clip in one process. Each TrajCrafter
        # holds DepthCrafter, the refinement model and (unless --prompt) BLIP-2,
        # so keeping one alive costs the next clip its memory.
        del crafter
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Rendered into {output_path.parent / output_path.stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
