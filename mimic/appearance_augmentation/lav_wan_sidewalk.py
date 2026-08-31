"""WAN-based sidewalk relighting with face-preserving mask blending.

Follows the original lav_wan_relight.py structure exactly for model loading
and inference (using pipe(...) directly), then applies face/person masks as
a post-hoc pixel-level blend at original resolution.

Usage:
    python lav_wan_sidewalk.py --video_path path/to/video.mp4 \
        [--save_dir output_sidewalk] [--n_lights 5] [--seed 42]

    # Skip auto face detection; provide pre-computed masks instead:
    python lav_wan_sidewalk.py --video_path path/to/video.mp4 \
        --mask_dir path/to/mask_folder/   # PNG masks, 255=preserve
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
import torch.nn.functional as F
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

# ── Prompt pool ──────────────────────────────────────────────────────
URBAN_RELIGHTING_PROMPTS = [
    "clear noon with direct sunlight",
    "sunny afternoon with strong shadows",
    "morning sunlight with soft illumination",
    "overcast morning with diffuse light",
    "overcast afternoon with gray uniform light",
    "partly cloudy sky with scattered sunlight",
    "summer midday with strong ground reflection",
    "winter low-angle sun with long shadows",
    "autumn afternoon with golden light",
    "urban canyon shadows between tall buildings",
    "sunset with warm orange glow",
    "blue hour after sunset with dim sky light",
    "pre-dawn with streetlights still on",
    "sunrise with low-angle warm light",
    "dusk with half-lit sky",
    "sunset reflection on glass facades",
    "morning fog with weak sunlight",
    "evening twilight with streetlights turning on",
    "early morning sun shining through skyscrapers",
    "winter sunrise with cold low-angle light",
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
    # ── additional diverse prompts ──
    "neon signs casting colorful reflections on wet pavement",
    "bright billboard illuminating a dark intersection",
    "hazy midday with washed-out diffuse sunlight",
    "dust-filled air scattering golden afternoon light",
    "thunderstorm with lightning flash illumination",
    "dense fog at night with glowing halos around lights",
    "moonlit street with faint blue ambient light",
    "parking garage with harsh fluorescent overhead lights",
    "underpass with mixed daylight and artificial ceiling lights",
    "construction site with portable floodlights at night",
    "fire hydrant spray creating rainbow refraction in sunlight",
    "spring morning with dappled light through tree canopy",
    "harsh midday desert sun with no shadows on open road",
    "early dawn with deep blue sky and faint horizon glow",
    "festival street with string lights and warm lanterns",
    "emergency vehicle flashing red and blue lights at night",
    "storefront awning casting sharp shadow on sidewalk",
]

BG_SOURCES = ["TOP", "RIGHT", "LEFT", "BOTTOM"]
BG_WEIGHTS = [0.2, 0.35, 0.35, 0.1]

TARGET_FPS = 15
DURATION_SEC = 8
WAN_NUM_FRAMES = TARGET_FPS * DURATION_SEC + 1  # 81 — (4k+1) for WAN VAE
OUTPUT_FRAMES = TARGET_FPS * DURATION_SEC        # 80


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
# Video I/O
# =====================================================================

def load_video_native(
    video_path: str, max_frames: int = WAN_NUM_FRAMES,
) -> tuple[list[np.ndarray], float]:
    """Return frames as uint8 numpy [H,W,3] at native resolution + fps."""
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    src_fps = meta.get("fps", TARGET_FPS)
    step = max(1, round(src_fps / TARGET_FPS))

    frames: list[np.ndarray] = []
    for idx, frame in enumerate(reader):
        if idx % step != 0:
            continue
        frames.append(frame)
        if len(frames) >= max_frames:
            break
    reader.close()
    return frames, src_fps


def frames_to_pil(
    frames: list[np.ndarray], width: int, height: int,
) -> list[Image.Image]:
    """Resize & centre-crop numpy frames → PIL list for the WAN pipe."""
    return [
        Image.fromarray(resize_and_center_crop(f, width, height))
        for f in frames
    ]


# =====================================================================
# Face / person mask generation
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
# Foreground-aware blending
# =====================================================================

def _extract_detail(frame: np.ndarray, ksize: int = 51) -> np.ndarray:
    """High-frequency detail layer: original minus its low-pass version."""
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
    """Blend relit video with original — upscale + detail transfer + fg anchor.

    If a neural upscaler is provided, it replaces cv2 resize and detail
    transfer is skipped (the upscaler already produces sharp output).
    """
    neural = upscaler is not None and upscale_method != "none"
    n = len(original_frames)
    out = []
    for i, (orig, mask) in enumerate(zip(original_frames, masks_orig)):
        H, W = orig.shape[:2]
        relit = relit_frames[i] if i < len(relit_frames) else relit_frames[-1]

        if neural:
            if (i + 1) % 10 == 0 or i == 0:
                print(f"    upscaling frame {i+1}/{n} …")
            relit_up = upscale_frame(relit, H, W, upscaler, upscale_method, prompt)
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


# =====================================================================
# Neural upscaling
# =====================================================================

def load_upscaler(method: str, device: str = "cuda"):
    """Load a neural super-resolution model.

    Returns an opaque upscaler object to pass into upscale_frame().
    """
    if method == "sd_x4":
        from diffusers import StableDiffusionUpscalePipeline

        up = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=torch.float16,
        )
        up = up.to(device)
        up.set_progress_bar_config(disable=True)
        return up

    if method == "realesrgan":
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        weight_dir = os.path.join(os.path.dirname(__file__), "weights")
        weight_path = os.path.join(weight_dir, "RealESRGAN_x4plus.pth")
        if not os.path.exists(weight_path):
            os.makedirs(weight_dir, exist_ok=True)
            download_url_to_file(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.1.0/RealESRGAN_x4plus.pth",
                weight_path,
            )
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                      num_block=23, num_grow_ch=32, scale=4)
        return RealESRGANer(
            scale=4, model_path=weight_path, model=net,
            half=True, device=device,
        )

    return None


def upscale_frame(
    frame: np.ndarray, target_h: int, target_w: int,
    upscaler, method: str, prompt: str = "",
) -> np.ndarray:
    """Upscale a single frame to exactly (target_h, target_w)."""
    h_in, w_in = frame.shape[:2]
    need_scale = max(target_h / h_in, target_w / w_in)

    if method == "sd_x4":
        pil_img = Image.fromarray(frame)
        with torch.no_grad():
            result = upscaler(
                prompt=prompt, image=pil_img,
                num_inference_steps=20, noise_level=20,
                guidance_scale=4.0,
            ).images[0]
        result = np.array(result)
        if result.shape[:2] != (target_h, target_w):
            result = cv2.resize(result, (target_w, target_h),
                                interpolation=cv2.INTER_LANCZOS4)
        return result

    if method == "realesrgan":
        outscale = min(max(need_scale, 1.0), 4.0)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output, _ = upscaler.enhance(bgr, outscale=outscale)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        if output.shape[:2] != (target_h, target_w):
            output = cv2.resize(output, (target_w, target_h),
                                interpolation=cv2.INTER_LANCZOS4)
        return output

    return cv2.resize(frame, (target_w, target_h),
                      interpolation=cv2.INTER_LANCZOS4)


# =====================================================================
# IC-Light pipeline loader (for single-image relighting, e.g. stylize_samples.py)
# =====================================================================

def load_ic_light_pipeline(
    device: str | torch.device = "cuda",
    *,
    sd_model: str = "stablediffusionapi/realistic-vision-v51",
    ic_light_model: str | None = None,
    gamma: float = 0.7,
    dtype: torch.dtype | None = None,
):
    """Load the IC-Light img2img pipeline (same models as this script's video relighting).

    Returns a pipeline that expects cross_attention_kwargs={'concat_conds': latent}
    when calling __call__(image=..., prompt=..., ...). The latent should be the
    VAE-encoded input image * vae.config.scaling_factor.
    """
    if dtype is None:
        dtype = torch.float16
    device = torch.device(device) if isinstance(device, str) else device
    if ic_light_model is None:
        ic_light_model = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", "iclight_sd15_fc.safetensors"
        )
    if not os.path.exists(ic_light_model):
        download_url_to_file(
            url="https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors",
            dst=ic_light_model,
        )

    ## SD modules for IC-Light
    tokenizer = CLIPTokenizer.from_pretrained(sd_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_model, subfolder="text_encoder")
    vae_sd = AutoencoderKL.from_pretrained(sd_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(sd_model, subfolder="unet")

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

    sd_offset = sf.load_file(ic_light_model)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged

    text_encoder = text_encoder.to(device=device, dtype=dtype)
    vae_sd = vae_sd.to(device=device, dtype=dtype)
    unet = unet.to(device=device, dtype=dtype)
    unet.set_attn_processor(AttnProcessor2_0())
    vae_sd.set_attn_processor(AttnProcessor2_0())

    @torch.inference_mode()
    def custom_forward_CLA(
        self, hidden_states, encoder_hidden_states=None,
        attention_mask=None, cross_attention_kwargs=None,
    ):
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
    ic_light_pipe = ic_light_pipe.to(device=device, dtype=dtype)
    ic_light_pipe.vae.requires_grad_(False)
    ic_light_pipe.unet.requires_grad_(False)
    return ic_light_pipe


# =====================================================================
# Main  — follows lav_wan_relight.py structure exactly
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WAN sidewalk relighting — face-preserving, original resolution.",
    )
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="output_sidewalk")
    parser.add_argument("--n_lights", type=int, default=3)
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Folder with pre-computed PNG masks (255=preserve). "
                             "If omitted, auto-detect faces via YOLO.")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt")

    parser.add_argument("--max_side", type=int, default=512,
                        help="Max side for model processing resolution.")
    parser.add_argument("--fg_preserve", type=float, default=0.3,
                        help="How much original foreground to blend in (0.0=fully relit, 0.3=recommended, 1.0=original).")
    parser.add_argument("--detail_strength", type=float, default=0.7,
                        help="High-freq detail transfer from original (0=blurry, 0.7=recommended, 1.0=full). "
                             "Ignored when --upscaler is set.")
    parser.add_argument("--upscaler", type=str, default="none",
                        choices=["none", "realesrgan", "sd_x4"],
                        help="Neural upscaler: 'realesrgan' (~50ms/frame, sharp), "
                             "'sd_x4' (diffusers SD x4, ~5s/frame, highest quality), "
                             "'none' (Lanczos + detail transfer).")
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--num_step", type=int, default=10)
    parser.add_argument("--text_guide_scale", type=float, default=1.5)
    parser.add_argument("--negative_prompt", type=str, default="bad quality, worse quality")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sd_model", type=str,
                        default="stablediffusionapi/realistic-vision-v51")
    parser.add_argument("--vdm_model", type=str,
                        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--ic_light_model", type=str,
                        default="./models/iclight_sd15_fc.safetensors")
    args = parser.parse_args()

    device = torch.device("cuda")
    adopted_dtype = torch.float16
    set_all_seed(args.seed)

    # ── 1. Load video at native resolution ────────────────────────────
    print(f"Loading video: {args.video_path}")
    orig_frames, src_fps = load_video_native(args.video_path, max_frames=WAN_NUM_FRAMES)
    actual = len(orig_frames)
    orig_h, orig_w = orig_frames[0].shape[:2]
    print(f"  {actual} frames at native {orig_w}×{orig_h}")

    if actual < WAN_NUM_FRAMES:
        orig_frames += [orig_frames[-1]] * (WAN_NUM_FRAMES - actual)

    model_w, model_h = compute_model_resolution(orig_w, orig_h, args.max_side)
    print(f"  Model resolution: {model_w}×{model_h}")

    # ── 2. Detect face / person masks at native resolution ────────────
    print("Generating face-preserve masks …")
    masks_orig = make_preserve_masks(
        orig_frames, mask_dir=args.mask_dir, yolo_model=args.yolo_model,
    )
    n_with_face = sum(1 for m in masks_orig if m.max() > 0)
    print(f"  {n_with_face}/{len(masks_orig)} frames have face/person regions")

    # ── 3. Prepare model-res PIL frames (same as read_video) ──────────
    video_list = frames_to_pil(orig_frames, model_w, model_h)

    # ── 4. Load models (identical to lav_wan_relight.py) ──────────────
    print("Loading models …")

    ## WAN VDM
    vae_wan = AutoencoderKLWan.from_pretrained(
        args.vdm_model, subfolder="vae", torch_dtype=adopted_dtype,
    )
    pipe = WanVideoToVideoPipeline.from_pretrained(
        args.vdm_model, vae=vae_wan, torch_dtype=adopted_dtype,
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    pipe = pipe.to(device=device, dtype=adopted_dtype)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    ## SD modules for IC-Light
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder")
    vae_sd = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet")

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

    ## IC-Light weights
    if not os.path.exists(args.ic_light_model):
        download_url_to_file(
            url="https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors",
            dst=args.ic_light_model,
        )
    sd_offset = sf.load_file(args.ic_light_model)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged

    text_encoder = text_encoder.to(device=device, dtype=adopted_dtype)
    vae_sd = vae_sd.to(device=device, dtype=adopted_dtype)
    unet = unet.to(device=device, dtype=adopted_dtype)
    unet.set_attn_processor(AttnProcessor2_0())
    vae_sd.set_attn_processor(AttnProcessor2_0())

    ## Consistent Light Attention (CLA)
    gamma = args.gamma

    @torch.inference_mode()
    def custom_forward_CLA(
        self, hidden_states, encoder_hidden_states=None,
        attention_mask=None, cross_attention_kwargs=None,
    ):
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

    ## IC-Light pipeline
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

    # ── 5. Random lights & run ────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    chosen_prompts = rng.choice(URBAN_RELIGHTING_PROMPTS, size=args.n_lights, replace=False)
    chosen_bg = rng.choice(BG_SOURCES, size=args.n_lights, p=BG_WEIGHTS)

    os.makedirs(args.save_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
    generator = torch.manual_seed(args.seed)
    vdm_prompt = "driving through urban sidewalk scenery"
    num_inference_steps = int(round(args.num_step / args.strength))

    use_neural_up = args.upscaler != "none"
    masks_trimmed = masks_orig[:OUTPUT_FRAMES]
    orig_trimmed = orig_frames[:OUTPUT_FRAMES]

    # Pre-load upscaler (lightweight models like Real-ESRGAN fit alongside)
    upscaler_obj = load_upscaler(args.upscaler, device=str(device)) if use_neural_up else None

    print(f"\n{'='*60}")
    print(f"Generating {args.n_lights} relit versions  ({model_w}×{model_h} → {orig_w}×{orig_h})")
    if use_neural_up:
        print(f"  Upscaler: {args.upscaler}")
    print(f"{'='*60}")

    for i, (prompt, bg_name) in enumerate(zip(chosen_prompts, chosen_bg)):
        bg_source = BGSource[bg_name]
        relight_prompt = f"sidewalk scene, {prompt}"
        print(f"\n[{i+1}/{args.n_lights}] prompt: \"{relight_prompt}\"  light: {bg_source.value}")

        with torch.no_grad():
            output = pipe(
                ic_light_pipe=ic_light_pipe,
                relight_prompt=relight_prompt,
                bg_source=bg_source,
                video=video_list,
                prompt=vdm_prompt,
                negative_prompt=args.negative_prompt,
                strength=args.strength,
                guidance_scale=args.text_guide_scale,
                num_inference_steps=num_inference_steps,
                height=model_h,
                width=model_w,
                num_frames=WAN_NUM_FRAMES,
                generator=generator,
            )
            relit_frames = (output.frames[0] * 255).astype(np.uint8)

        relit_trimmed = relit_frames[:OUTPUT_FRAMES]

        blended = blend_with_original(
            orig_trimmed, relit_trimmed, masks_trimmed,
            fg_preserve=args.fg_preserve,
            detail_strength=args.detail_strength,
            upscaler=upscaler_obj,
            upscale_method=args.upscaler,
            prompt=relight_prompt,
        )

        prompt_slug = prompt.replace(" ", "_").replace(",", "")[:60]
        save_name = f"{video_stem}_relight_{i:02d}_{prompt_slug}.mp4"
        save_path = os.path.join(args.save_dir, save_name)
        imageio.mimwrite(save_path, blended, fps=TARGET_FPS)
        print(f"  Saved {len(blended)} frames @ {orig_w}×{orig_h} → {save_path}")

    if upscaler_obj is not None:
        del upscaler_obj
        torch.cuda.empty_cache()

    print(f"\nDone! {args.n_lights} relit videos saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
