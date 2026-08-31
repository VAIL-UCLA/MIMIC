"""Video randomization for urban sidewalk navigation: one full video, one light per 8s.

Batch-level flow:
  1. Full video → m segments (8s each).
  2. Sample m prompts (and m bg sources) once — segment i uses prompt i % n_lights.
  3. Build a flat list of chunk jobs: (segment_idx, chunk_start, prompt, bg).
  4. Process jobs (by segment to avoid reloading): load segment, run all its chunk jobs, then next segment.
  5. Reassemble: trim per-segment results and concatenate → one full-length video.

Saves: {stem}_relight_full.mp4 (same resolution and fps as input).

Speed: --fast (lower res, fewer steps); --compile (PyTorch 2+).

Usage:
    python lav_randomize_video.py --video_path path/to/video.mp4 \\
        [--save_dir output_random] [--n_lights 5] [--fast] [--compile] [--seed 42]
"""

import os
import argparse
import glob
import random
from types import MethodType

import cv2
import imageio
import numpy as np
import torch
import safetensors.torch as sf
from PIL import Image
from torch.hub import download_url_to_file

from diffusers import (
    AutoencoderKL,
    AutoencoderKLWan,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

from src.ic_light import BGSource
from src.ic_light_pipe import StableDiffusionImg2ImgPipeline
from src.wan_pipe import WanVideoToVideoPipeline
from utils.tools import set_all_seed, resize_and_center_crop

# Reuse upscaler from lav_wan_sidewalk when --upscaler is set
try:
    from lav_wan_sidewalk import load_upscaler as _load_upscaler_from_sidewalk, upscale_frame as _upscale_frame_sidewalk
except Exception:
    _load_upscaler_from_sidewalk = None
    _upscale_frame_sidewalk = None

# ── WAN model frame constraint: (4k+1) for VAE; we pad/trim to this for inference
WAN_MODEL_NUM_FRAMES = 81  # 80 output frames after pipeline (model uses 81 internally)

# ── Segment duration for each iteration (seconds); each segment saved as _start_{index}
SEGMENT_DURATION_SEC = 8

# ── Prompt pool: lighting / weather / time-of-day / environment variations
# Each prompt describes a lighting condition — the IC-Light model re-renders the
# scene under this condition while the WAN VDM preserves temporal consistency.

URBAN_RELIGHTING_PROMPTS = [
    # ── Daytime / Sun positions ──
    "clear noon with direct sunlight",
    "sunny afternoon with strong shadows",
    "morning sunlight with soft illumination",
    "early morning golden hour with long warm shadows",
    "late afternoon sun casting long diagonal shadows",
    "midday sun with harsh overhead lighting and minimal shadows",
    "high noon sun with bleached highlights and deep shadows",
    "afternoon sun partially blocked by clouds",
    "bright midday with strong specular highlights on pavement",
    "early afternoon with warm side-lighting from the west",
    # ── Overcast / Cloudy ──
    "overcast morning with diffuse light",
    "overcast afternoon with gray uniform light",
    "partly cloudy sky with scattered sunlight",
    "thick overcast with flat shadowless lighting",
    "thin cloud cover with soft diffuse sunlight",
    "bright overcast day with even illumination everywhere",
    "heavy cloud cover creating low-contrast muted colors",
    "silver overcast sky reflecting off wet surfaces",
    # ── Seasons ──
    "summer midday with strong ground reflection",
    "winter low-angle sun with long shadows",
    "autumn afternoon with golden light",
    "spring morning with fresh bright light through scattered clouds",
    "winter sunrise with cold low-angle light",
    "late autumn dusk with warm amber tones and bare trees",
    "midsummer evening with extended golden hour",
    "early spring overcast with cool blue-white light",
    # ── Urban shadows ──
    "urban canyon shadows between tall buildings",
    "shadows from overhead bridge or overpass",
    "building shadow edge with half sun half shade",
    "narrow alley with only sky-reflected ambient light",
    "deep urban canyon with light only reaching ground at noon",
    "patchy shadows from trees lining the sidewalk",
    "dappled sunlight filtering through overhead tree canopy",
    # ── Sunset / Sunrise / Twilight ──
    "sunset with warm orange glow",
    "blue hour after sunset with dim sky light",
    "pre-dawn with streetlights still on",
    "sunrise with low-angle warm light",
    "dusk with half-lit sky",
    "sunset reflection on glass facades",
    "evening twilight with streetlights turning on",
    "early morning sun shining through skyscrapers",
    "deep orange sunset with silhouetted buildings",
    "pink and purple twilight sky after sunset",
    "civil twilight with balanced ambient and artificial light",
    "nautical twilight with deep blue sky and first stars",
    "golden hour backlighting with rim-lit pedestrians",
    "sunrise through morning haze with warm diffused beams",
    # ── Night / Artificial light ──
    "streetlights with sodium vapor orange glow",
    "streetlights with white LED illumination",
    "commercial street at night with shop signs lit",
    "traffic light illuminating stopped vehicles",
    "car headlights creating bright road patches",
    "red car taillights reflecting on road",
    "pedestrians crossing under mixed street and car lights",
    "highway at night with only car headlights",
    "residential street with warm yellow lights",
    "tunnel interior with evenly spaced lights",
    "moonlit night with cool blue ambient light",
    "full moon illuminating an empty street",
    "night with only distant city glow on horizon",
    "parking lot at night with overhead fluorescent lights",
    "gas station canopy with bright white lighting at night",
    "bus stop shelter lit by cold white light at night",
    "neon signs casting colored light on wet pavement",
    "stadium lighting spilling onto nearby streets",
    "construction zone with harsh temporary flood lights",
    "underpass lit by amber sodium lights and passing headlights",
    "dark alley with single overhead bulb",
    "nighttime crosswalk illuminated by dedicated crossing light",
    "emergency vehicle flashing red and blue lights on scene",
    # ── Rain ──
    "light rain with slightly reflective road surface",
    "heavy rain with headlights reflecting on wet road",
    "downpour with low visibility under streetlights",
    "after rain with wet ground reflections",
    "wet road at night reflecting street and car lights",
    "continuous rainy day with gray low contrast lighting",
    "pedestrians with umbrellas in scattered rain light",
    "fog combined with rain scattering car lights",
    "highway in rain and mist with blurred taillights",
    "rain streaks on glass facades reflecting light",
    "light drizzle with glistening road surface in afternoon",
    "thunderstorm with dark sky and intermittent lightning flash",
    "monsoon heavy rain with near-zero visibility",
    "post-rain golden hour with puddle reflections of sunset",
    # ── Snow / Ice ──
    "daytime snow with bright ground reflection",
    "nighttime snow reflecting streetlights",
    "snowfall reducing visibility",
    "morning sunlight shining on snowy streets",
    "headlights forming beams through falling snow",
    "icy road with strong reflections from frozen surface",
    "sidewalks covered in snow with diffuse light",
    "melting snow with mixed water and ice reflections",
    "daytime snow fog with heavy light scattering",
    "nighttime icy road with taillights reflecting on ice",
    "blizzard whiteout with minimal visibility",
    "fresh snow on ground reflecting bright blue sky",
    "sleet with mixed rain and ice on road surface",
    # ── Fog / Haze / Mist ──
    "morning fog with weak sunlight",
    "dense fog with visibility under 50 meters",
    "thin morning mist with sun breaking through",
    "coastal fog rolling over urban streets",
    "industrial haze reducing contrast and color saturation",
    "smog with yellowish haze and reduced visibility",
    "early morning ground fog with clear sky above",
    "nighttime fog with halos around streetlights",
    "fog bank with sharp visibility boundary",
    "summer heat haze creating shimmering road surface",
    # ── Dust / Sand ──
    "dust storm with orange-tinted reduced visibility",
    "light dust haze with warm tinted sunlight",
    "sandy wind with particles catching sunlight",
    # ── Indoor-like / Covered ──
    "covered walkway with overhead artificial lighting",
    "parking garage interior with fluorescent ceiling lights",
    "shopping arcade with mixed daylight and shop lighting",
    "train station platform under roof with skylight panels",
    "pedestrian underpass with cold fluorescent tubes",
    # ── Camera / Sensor ──
    "slightly overexposed bright scene with blown highlights",
    "underexposed dark scene with visible noise",
    "high dynamic range scene with bright sky and dark ground",
    "backlit scene with strong lens flare from sun",
    "low-light scene with visible camera sensor noise",
    "motion blur from fast camera movement in low light",
    # ── Geographic / Environmental ──
    "tropical midday with harsh equatorial sun and deep shadows",
    "northern latitude winter with sun barely above horizon",
    "desert environment with intense direct sun and clear sky",
    "Mediterranean afternoon with warm golden light and blue shadows",
    "Pacific Northwest overcast with soft green-tinted light",
    "Southeast Asian monsoon season overcast with warm humid haze",
    "Nordic summer midnight sun with low warm light from north",
    "Australian outback harsh midday sun with red earth reflection",
    # ── Mixed / Complex ──
    "transitional light from sunshine to sudden cloud cover",
    "mixed indoor-outdoor light at building entrance",
    "sunrise with morning fog creating light rays through trees",
    "night rain with neon shop signs reflecting on wet street",
    "snow falling under orange sodium streetlights",
    "sunset through industrial smoke creating dramatic red sky",
    "bright cloudy day with occasional sun breaks through gaps",
    "late evening with last sunlight on building tops and shadows below",
]

# Simulator-style prompts as normal pool entries (chosen at random like any other)
SIMULATOR_STYLE_PROMPTS = [
    "Isaac Sim simulator rendering style, photorealistic RTX, urban sidewalk navigation, clear noon with direct sunlight",
    "Isaac Sim simulator rendering style, photorealistic RTX, urban sidewalk navigation, overcast afternoon with gray uniform light",
    "Isaac Sim simulator rendering style, photorealistic RTX, urban sidewalk navigation, wet road at night with neon reflections",
    "Isaac Sim simulator rendering style, photorealistic RTX, urban sidewalk navigation, golden hour sunset",
    "CARLA simulator rendering style, urban sidewalk navigation, sunny afternoon with strong shadows",
    "CARLA simulator rendering style, urban sidewalk navigation, streetlights with sodium vapor orange glow",
    "CARLA simulator rendering style, urban sidewalk navigation, heavy rain at night",
    "CARLA simulator rendering style, urban sidewalk navigation, foggy morning with low visibility",
    "MetaDrive MetaUrban style, urban sidewalk navigation, overcast afternoon with gray uniform light",
    "MetaDrive MetaUrban style, urban sidewalk navigation, dusk with half-lit sky",
    "MetaDrive MetaUrban style, urban sidewalk navigation, bright noon with harsh shadows",
    "Unreal Engine 5 Lumen rendering, photorealistic urban sidewalk, clear noon with direct sunlight",
    "Unreal Engine 5 Lumen rendering, photorealistic urban sidewalk, blue hour after sunset with dim sky light",
    "Unreal Engine 5 Lumen rendering, photorealistic urban sidewalk, rainy night with puddle reflections",
    "Unreal Engine 5 Lumen rendering, photorealistic urban sidewalk, snowy winter morning",
    "Unity HDRP simulator style, urban pedestrian path, morning sunlight with soft illumination",
    "Unity HDRP simulator style, urban pedestrian path, tunnel interior with evenly spaced lights",
    "Unity HDRP simulator style, urban pedestrian path, dense urban night with mixed lighting",
    "LGSVL simulator rendering, urban sidewalk navigation, sunny afternoon with strong shadows",
    "LGSVL simulator rendering, urban sidewalk navigation, wet road at night reflecting street and car lights",
    "NVIDIA Drive Sim rendering style, urban street scene, overcast morning with diffuse light",
    "NVIDIA Drive Sim rendering style, urban street scene, commercial street at night with shop signs lit",
    "NVIDIA Drive Sim rendering style, urban street scene, highway at dusk with headlights",
    "OpenPCDet-style LiDAR simulation, urban sidewalk, clear noon with direct sunlight",
    "SynthCity synthetic urban dataset style, sidewalk navigation, winter low-angle sun with long shadows",
    "GTA-V style urban rendering, pedestrian view, sunset with warm orange glow",
    "GTA-V style urban rendering, pedestrian view, rainy night downtown with neon",
    "GTA-V style urban rendering, pedestrian view, foggy morning coastal road",
    "Waymo Open Dataset simulator style, urban sidewalk, morning fog with weak sunlight",
    "Waymo Open Dataset simulator style, urban sidewalk, bright afternoon suburban street",
    "AirSim simulator rendering, urban drone view, clear sky with strong sunlight",
    "nuScenes dataset style, driving camera view, nighttime urban intersection",
    "nuScenes dataset style, driving camera view, rainy highway with spray from vehicles",
    "Habitat simulator style, indoor-outdoor transition, bright exterior dark interior",
]


def build_relight_prompt_pool():
    """Build one flat prompt pool: base lighting prompts + simulator options in the same random pool."""
    pool = ["sidewalk scene, " + p for p in URBAN_RELIGHTING_PROMPTS]
    pool.extend(SIMULATOR_STYLE_PROMPTS)
    return pool


RELIGHT_PROMPT_POOL = build_relight_prompt_pool()

BG_SOURCES = ["TOP", "RIGHT", "LEFT", "BOTTOM"]
BG_WEIGHTS = [0.2, 0.35, 0.35, 0.1]


# =====================================================================
# Resolution helpers
# =====================================================================

def _round16(x: int) -> int:
    return max(16, round(x / 16) * 16)


def compute_model_resolution(
    orig_w: int, orig_h: int, max_side: int = 480,
) -> tuple[int, int]:
    """Scale to fit *max_side*, then round each dim to nearest multiple of 16."""
    scale = min(max_side / max(orig_w, orig_h), 1.0)
    return _round16(int(orig_w * scale)), _round16(int(orig_h * scale))


# =====================================================================
# Video I/O — preserve original frame count, resolution, and fps
# =====================================================================

def get_video_metadata(video_path: str) -> tuple[int, float, int, int]:
    """Get total frame count, fps, width, height without loading all frames."""
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    src_fps = meta.get("fps", 10.0)
    count = 0
    first_frame = None
    for frame in reader:
        if first_frame is None:
            first_frame = frame
        count += 1
    reader.close()
    if first_frame is None:
        raise ValueError(f"No frames in {video_path}")
    orig_h, orig_w = first_frame.shape[:2]
    return count, src_fps, orig_w, orig_h


def load_video_segment(
    video_path: str,
    start_frame_idx: int,
    num_frames: int,
) -> tuple[list[np.ndarray], float, int, int]:
    """Load a segment of video: frames [start_frame_idx, start_frame_idx+num_frames).

    Returns:
        frames: list of uint8 [H,W,3]
        src_fps: original fps
        orig_w, orig_h: native width and height
    """
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    src_fps = meta.get("fps", 10.0)
    frames: list[np.ndarray] = []
    for idx, frame in enumerate(reader):
        if idx < start_frame_idx:
            continue
        frames.append(frame)
        if len(frames) >= num_frames:
            break
    reader.close()
    if not frames:
        raise ValueError(f"No frames read from {video_path} segment start={start_frame_idx}")
    orig_h, orig_w = frames[0].shape[:2]
    return frames, src_fps, orig_w, orig_h


def load_video_preserve_native(
    video_path: str, max_frames: int = WAN_MODEL_NUM_FRAMES,
) -> tuple[list[np.ndarray], float, int, int]:
    """Load video at native resolution and fps. No temporal subsampling.

    Returns:
        frames: list of uint8 [H,W,3] (length <= max_frames)
        src_fps: original fps
        orig_w, orig_h: native width and height
    """
    return load_video_segment(video_path, 0, max_frames)


def frames_to_pil(
    frames: list[np.ndarray], width: int, height: int,
) -> list[Image.Image]:
    """Resize & centre-crop numpy frames → PIL list for the WAN pipe."""
    return [
        Image.fromarray(resize_and_center_crop(f, width, height))
        for f in frames
    ]


# =====================================================================
# Face / person mask (same as lav_wan_sidewalk)
# =====================================================================

def detect_faces_yolo(
    frames: list[np.ndarray],
    model_name: str = "yolov8n.pt",
    person_conf: float = 0.30,
    dilate_px: int = 30,
    blur_sigma: int = 21,
) -> list[np.ndarray]:
    """Detect *person* bboxes via YOLO → soft float32 masks [H,W], 1=preserve."""
    from ultralytics import YOLO

    detector = YOLO(model_name)
    H, W = frames[0].shape[:2]
    masks = []

    for frame in frames:
        results = detector(frame, verbose=False, conf=person_conf)
        mask = np.zeros((H, W), dtype=np.float32)

        for r in results:
            if r.boxes is None:
                continue
            for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                if int(cls) != 0:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                head_y2 = y1 + int((y2 - y1) * 0.4)
                x1d = max(0, x1 - dilate_px)
                y1d = max(0, y1 - dilate_px)
                x2d = min(W, x2 + dilate_px)
                y2d = min(H, head_y2 + dilate_px)
                mask[y1d:y2d, x1d:x2d] = 1.0

        if mask.max() > 0:
            mask = cv2.GaussianBlur(mask, (blur_sigma, blur_sigma), 0)
            mask = np.clip(mask, 0.0, 1.0)
        masks.append(mask)
    return masks


def load_masks_from_dir(mask_dir: str, num_frames: int) -> list[np.ndarray]:
    """Load pre-computed masks (PNG, 255=preserve) → float32 [H,W] in [0,1]."""
    paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in paths[:num_frames]:
        m = np.array(Image.open(p).convert("L")).astype(np.float32) / 255.0
        masks.append(m)
    while len(masks) < num_frames:
        masks.append(masks[-1] if masks else np.zeros((64, 64), dtype=np.float32))
    return masks


def make_preserve_masks(
    frames: list[np.ndarray],
    mask_dir: str | None = None,
    yolo_model: str = "yolov8n.pt",
) -> list[np.ndarray]:
    """Return per-frame float32 masks [H,W] where 1=preserve, 0=relight."""
    if mask_dir and os.path.isdir(mask_dir):
        print(f"  Loading masks from {mask_dir}")
        return load_masks_from_dir(mask_dir, len(frames))

    try:
        print("  Auto-detecting faces/people with YOLO …")
        return detect_faces_yolo(frames, model_name=yolo_model)
    except Exception as e:
        print(f"  YOLO detection failed ({e}), trying cv2 cascade …")

    try:
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml",
        )
        face_cascade = cv2.CascadeClassifier(cascade_path)
        H, W = frames[0].shape[:2]
        masks = []
        dilate = 40
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            mask = np.zeros((H, W), dtype=np.float32)
            for (x, y, w, h) in faces:
                x1 = max(0, x - dilate)
                y1 = max(0, y - dilate)
                x2 = min(W, x + w + dilate)
                y2 = min(H, y + h + dilate)
                mask[y1:y2, x1:x2] = 1.0
            if mask.max() > 0:
                mask = cv2.GaussianBlur(mask, (21, 21), 0)
                mask = np.clip(mask, 0.0, 1.0)
            masks.append(mask)
        return masks
    except Exception as e2:
        print(f"  cv2 cascade also failed ({e2}). No face mask — full relighting.")
        H, W = frames[0].shape[:2]
        return [np.zeros((H, W), dtype=np.float32)] * len(frames)


# =====================================================================
# Foreground-aware blending (same as lav_wan_sidewalk)
# =====================================================================

def _extract_detail(frame: np.ndarray, ksize: int = 51) -> np.ndarray:
    low = cv2.GaussianBlur(frame.astype(np.float32), (ksize, ksize), 0)
    return frame.astype(np.float32) - low


def blend_with_original(
    original_frames: list[np.ndarray],
    relit_frames: np.ndarray,
    masks_orig: list[np.ndarray],
    fg_preserve: float = 0.3,
    detail_strength: float = 0.7,
    mask_blur_ksize: int = 21,
    upscaler=None,
    upscale_method: str = "none",
    prompt: str = "",
) -> list[np.ndarray]:
    neural = upscaler is not None and upscale_method != "none"
    n = len(original_frames)
    out = []
    for i, (orig, mask) in enumerate(zip(original_frames, masks_orig)):
        H, W = orig.shape[:2]
        relit = relit_frames[i] if i < len(relit_frames) else relit_frames[-1]

        if neural and _upscale_frame_sidewalk is not None:
            if (i + 1) % 10 == 0 or i == 0:
                print(f"    upscaling frame {i+1}/{n} …")
            relit_up = _upscale_frame_sidewalk(relit, H, W, upscaler, upscale_method, prompt)
        else:
            relit_up = cv2.resize(relit, (W, H), interpolation=cv2.INTER_LANCZOS4)

        result = relit_up.astype(np.float32)

        if not neural and detail_strength > 0.0:
            hf = _extract_detail(orig)
            result = result + detail_strength * hf

        if mask.max() > 0.01 and fg_preserve > 0.0:
            soft = cv2.GaussianBlur(mask, (mask_blur_ksize, mask_blur_ksize), 0)
            alpha = np.clip(soft, 0.0, 1.0)[..., None]
            result = result + alpha * fg_preserve * (
                orig.astype(np.float32) - result
            )

        out.append(np.clip(result, 0, 255).astype(np.uint8))
    return out


def load_upscaler(method: str, device: str = "cuda"):
    """Load neural upscaler; delegates to lav_wan_sidewalk if available."""
    if _load_upscaler_from_sidewalk is not None:
        return _load_upscaler_from_sidewalk(method, device)
    return None


# =====================================================================
# Pipeline state and process_one_video (for batch: load once, process many)
# =====================================================================

class PipelineState:
    """Holds loaded models and config for process_one_video."""
    pass


def load_pipeline(args) -> PipelineState:
    """Load models once. Returns state to pass to process_one_video(state, video_path, output_path)."""
    device = torch.device("cuda")
    adopted_dtype = torch.float16
    set_all_seed(args.seed)

    state = PipelineState()
    state.device = device
    state.adopted_dtype = adopted_dtype
    state.max_side = getattr(args, "max_side", 480)
    state.n_lights = getattr(args, "n_lights", 5)
    state.seed = getattr(args, "seed", 42)
    state.yolo_model = getattr(args, "yolo_model", "yolov8n.pt")
    state.fg_preserve = getattr(args, "fg_preserve", 0.3)
    state.detail_strength = getattr(args, "detail_strength", 0.7)
    state.upscaler = getattr(args, "upscaler", "none")
    state.negative_prompt = getattr(args, "negative_prompt", "bad quality, worse quality")
    state.strength = getattr(args, "strength", 0.35)
    state.num_step = getattr(args, "num_step", 10)
    state.text_guide_scale = getattr(args, "text_guide_scale", 1.5)
    state.gamma = getattr(args, "gamma", 0.7)
    state.no_yolo = getattr(args, "no_yolo", False)
    state.sd_model = getattr(args, "sd_model", "stablediffusionapi/realistic-vision-v51")
    state.vdm_model = getattr(args, "vdm_model", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    state.ic_light_model = getattr(args, "ic_light_model", "./models/iclight_sd15_fc.safetensors")
    state.do_compile = getattr(args, "compile", False)
    state.num_for_model = WAN_MODEL_NUM_FRAMES

    print("Loading models …")
    vae_wan = AutoencoderKLWan.from_pretrained(
        state.vdm_model, subfolder="vae", torch_dtype=adopted_dtype,
    )
    pipe = WanVideoToVideoPipeline.from_pretrained(
        state.vdm_model, vae=vae_wan, torch_dtype=adopted_dtype,
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    pipe = pipe.to(device=device, dtype=adopted_dtype)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    if state.do_compile and hasattr(torch, "compile"):
        print("Compiling VDM transformer …")
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
    state.pipe = pipe

    tokenizer = CLIPTokenizer.from_pretrained(state.sd_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(state.sd_model, subfolder="text_encoder")
    vae_sd = AutoencoderKL.from_pretrained(state.sd_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(state.sd_model, subfolder="unet")

    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(
            8, unet.conv_in.out_channels,
            unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding,
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    unet_original_forward = unet.forward

    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs["cross_attention_kwargs"] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

    unet.forward = hooked_unet_forward

    if not os.path.exists(state.ic_light_model):
        download_url_to_file(
            url="https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors",
            dst=state.ic_light_model,
        )
    sd_offset = sf.load_file(state.ic_light_model)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged

    text_encoder = text_encoder.to(device=device, dtype=adopted_dtype)
    vae_sd = vae_sd.to(device=device, dtype=adopted_dtype)
    unet = unet.to(device=device, dtype=adopted_dtype)
    unet.set_attn_processor(AttnProcessor2_0())
    vae_sd.set_attn_processor(AttnProcessor2_0())

    gamma = state.gamma

    @torch.inference_mode()
    def custom_forward_CLA(
        self, hidden_states, encoder_hidden_states=None,
        attention_mask=None, cross_attention_kwargs=None,
    ):
        import torch.nn.functional as F
        batch_size, sequence_length, channel = hidden_states.shape
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        shape = query.shape
        mean_key = (
            key.reshape(2, -1, shape[1], shape[2], shape[3])
            .mean(dim=1, keepdim=True)
            .expand(-1, shape[0] // 2, -1, -1, -1)
            .reshape(shape[0], shape[1], shape[2], shape[3])
        )
        mean_value = (
            value.reshape(2, -1, shape[1], shape[2], shape[3])
            .mean(dim=1, keepdim=True)
            .expand(-1, shape[0] // 2, -1, -1, -1)
            .reshape(shape[0], shape[1], shape[2], shape[3])
        )
        hidden_states_mean = F.scaled_dot_product_attention(
            query, mean_key, mean_value, attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        hidden_states = (1 - gamma) * hidden_states + gamma * hidden_states_mean
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if self.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / self.rescale_output_factor
        return hidden_states

    @torch.inference_mode()
    def prep_unet_self_attention(unet_model):
        for name, module in unet_model.named_modules():
            module_name = type(module).__name__
            parts = name.split(".")
            if ("Attention" in module_name
                    and parts[0] in "up_blocks"
                    and parts[-1] in ("attn1",)
                    and parts[1] not in "3"):
                module.forward = MethodType(custom_forward_CLA, module)
        return unet_model

    unet = prep_unet_self_attention(unet)

    ic_light_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085, beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True, steps_offset=1,
    )
    ic_light_pipe = StableDiffusionImg2ImgPipeline(
        vae=vae_sd, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=ic_light_scheduler,
        safety_checker=None, requires_safety_checker=False,
        feature_extractor=None, image_encoder=None,
    )
    ic_light_pipe = ic_light_pipe.to(device=device, dtype=adopted_dtype)
    ic_light_pipe.vae.requires_grad_(False)
    ic_light_pipe.unet.requires_grad_(False)
    state.ic_light_pipe = ic_light_pipe

    state.upscaler_obj = load_upscaler(state.upscaler, device=str(device)) if state.upscaler != "none" else None
    state.vdm_prompt = "urban sidewalk navigation, walking along pedestrian path"
    return state


def process_one_video(state: PipelineState, video_path: str, output_path: str) -> bool:
    """Process a single video and write the relit full video to output_path. Returns True on success."""
    total_frames, src_fps, orig_w, orig_h = get_video_metadata(video_path)
    frames_per_segment = max(1, round(SEGMENT_DURATION_SEC * src_fps))
    num_segments = (total_frames + frames_per_segment - 1) // frames_per_segment
    model_w, model_h = compute_model_resolution(orig_w, orig_h, state.max_side)

    rng = random.Random(state.seed)
    chosen_prompts = rng.sample(RELIGHT_PROMPT_POOL, min(state.n_lights, len(RELIGHT_PROMPT_POOL)))
    chosen_bg = rng.choices(BG_SOURCES, weights=BG_WEIGHTS, k=max(num_segments, len(chosen_prompts)))

    chunk_size = WAN_MODEL_NUM_FRAMES
    jobs: list[tuple[int, int, str, BGSource]] = []
    for seg_idx in range(num_segments):
        start_f = seg_idx * frames_per_segment
        segment_len = min(frames_per_segment, total_frames - start_f)
        if segment_len <= 0:
            break
        prompt = chosen_prompts[seg_idx % len(chosen_prompts)]
        bg_source = BGSource[chosen_bg[seg_idx]]
        chunk_start = 0
        while chunk_start < segment_len:
            jobs.append((seg_idx, chunk_start, prompt, bg_source))
            chunk_start += chunk_size

    generator = torch.manual_seed(state.seed)
    num_inference_steps = int(round(state.num_step / state.strength))

    seg_results: list[list[np.ndarray]] = [[] for _ in range(num_segments)]
    current_seg = -1
    seg_frames: list[np.ndarray] = []

    for seg_idx, chunk_start, relight_prompt, bg_source in jobs:
        if seg_idx != current_seg:
            current_seg = seg_idx
            start_f = seg_idx * frames_per_segment
            segment_len = min(frames_per_segment, total_frames - start_f)
            seg_frames, seg_fps, seg_w, seg_h = load_video_segment(
                video_path, start_f, segment_len,
            )
            assert seg_fps == src_fps and seg_w == orig_w and seg_h == orig_h

        chunk_frames = seg_frames[chunk_start : chunk_start + chunk_size]
        if len(chunk_frames) < chunk_size:
            chunk_frames = chunk_frames + [chunk_frames[-1]] * (chunk_size - len(chunk_frames))

        if getattr(state, "no_yolo", False):
            H, W = chunk_frames[0].shape[:2]
            masks_chunk = [np.zeros((H, W), dtype=np.float32)] * len(chunk_frames)
        else:
            masks_chunk = make_preserve_masks(
                chunk_frames, mask_dir=None, yolo_model=state.yolo_model,
            )
        video_list = frames_to_pil(chunk_frames, model_w, model_h)

        with torch.no_grad():
            output = state.pipe(
                ic_light_pipe=state.ic_light_pipe,
                relight_prompt=relight_prompt,
                bg_source=bg_source,
                video=video_list,
                prompt=state.vdm_prompt,
                negative_prompt=state.negative_prompt,
                strength=state.strength,
                guidance_scale=state.text_guide_scale,
                num_inference_steps=num_inference_steps,
                height=model_h,
                width=model_w,
                num_frames=state.num_for_model,
                generator=generator,
            )
            relit_np = (output.frames[0] * 255).astype(np.uint8)

        num_out = min(len(chunk_frames), len(relit_np))
        blended = blend_with_original(
            chunk_frames[:num_out], relit_np[:num_out], masks_chunk[:num_out],
            fg_preserve=state.fg_preserve,
            detail_strength=state.detail_strength,
            upscaler=state.upscaler_obj,
            upscale_method=state.upscaler,
            prompt=relight_prompt,
        )
        seg_results[seg_idx].extend(blended)

    full_frames = []
    for seg_idx in range(num_segments):
        start_f = seg_idx * frames_per_segment
        segment_len = min(frames_per_segment, total_frames - start_f)
        if segment_len <= 0:
            break
        full_frames.extend(seg_results[seg_idx][:segment_len])

    if not full_frames:
        return False
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    imageio.mimwrite(output_path, full_frames, fps=src_fps)
    return True


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Video randomization for urban sidewalk navigation — same frames, resolution, fps as input.",
    )
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="output_random")
    parser.add_argument("--n_lights", type=int, default=5,
                        help="Number of relit variants to generate.")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Folder with pre-computed PNG masks (255=preserve).")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt")

    parser.add_argument("--max_side", type=int, default=None,
                        help="Max side for model resolution (default: 480, or 384 if --fast).")
    parser.add_argument("--fast", action="store_true",
                        help="Faster run: lower resolution (384), fewer steps (6), higher strength (0.45).")
    parser.add_argument("--fg_preserve", type=float, default=0.3)
    parser.add_argument("--detail_strength", type=float, default=0.7)
    parser.add_argument("--upscaler", type=str, default="none",
                        choices=["none", "realesrgan", "sd_x4"])
    parser.add_argument("--strength", type=float, default=None,
                        help="Relight strength (default: 0.35, or 0.45 if --fast).")
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--num_step", type=int, default=None,
                        help="Effective steps scale (default: 10, or 6 if --fast); actual steps = round(num_step/strength).")
    parser.add_argument("--text_guide_scale", type=float, default=1.0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "bad quality, worse quality, low quality, low resolution, blurry, blur, out of focus, "
            "distorted, deformed, disfigured, ugly, bad anatomy, wrong proportions, "
            "oversaturated, underexposed, overexposed, flat lighting, harsh shadows, "
            "artifacts, noise, grain, watermark, text, logo, duplicate, mutilated, "
            "cropped, jpeg artifacts, compression artifacts, flickering, inconsistent lighting"
        ),
    )
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile on the VDM pipeline (PyTorch 2+). Faster after first chunk.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sd_model", type=str,
                        default="stablediffusionapi/realistic-vision-v51")
    parser.add_argument("--vdm_model", type=str,
                        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--ic_light_model", type=str,
                        default="./models/iclight_sd15_fc.safetensors")
    args = parser.parse_args()

    # --fast: lower resolution, fewer steps, higher strengzth (faster inference)
    if args.fast:
        if args.max_side is None:
            args.max_side = 512
        if args.strength is None:
            args.strength = 0.25
        if args.num_step is None:
            args.num_step = 6
        print("Using --fast: lower resolution, fewer steps.")
    if args.max_side is None:
        args.max_side = 480
    if args.strength is None:
        args.strength = 0.35
    if args.num_step is None:
        args.num_step = 10

    state = load_pipeline(args)
    video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
    os.makedirs(args.save_dir, exist_ok=True)
    output_path = os.path.join(args.save_dir, f"{video_stem}_relight_full.mp4")
    print(f"Processing: {args.video_path} → {output_path}")
    ok = process_one_video(state, args.video_path, output_path)
    if state.upscaler_obj is not None:
        del state.upscaler_obj
        torch.cuda.empty_cache()
    if ok:
        print(f"Done! Saved to {output_path}")
    else:
        print("No frames produced.")


if __name__ == "__main__":
    main()
