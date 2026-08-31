"""AnimateDiff-based sidewalk relighting with background inpainting.

Based on lav_paint.py. For each lighting condition, produces TWO samples:
  1. "inpaint" — relit foreground + generated background (full pipeline)
  2. "replace" — generated background + ORIGINAL foreground pasted back

Uses foreground masks from extract_foreground.py (pedestrians + vehicles).

Usage:
    python lav_paint_sidewalk.py --video_path path/to/video.mp4 \
        --mask_dir path/to/masks/ [--save_dir output_paint] [--n_lights 5]
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
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    MotionAdapter,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

from src.ic_light import BGSource, Relighter
from src.ic_light_pipe import StableDiffusionImg2ImgPipeline
from src.animatediff_inpaint_pipe import AnimateDiffVideoToVideoPipeline
from utils.tools import set_all_seed, read_video, read_mask, get_fg_video, resize_and_center_crop

# ── Prompt pool (same as lav_wan_sidewalk.py) ────────────────────────
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
]

BG_SOURCES = ["TOP", "RIGHT", "LEFT", "BOTTOM"]
BG_WEIGHTS = [0.2, 0.35, 0.35, 0.1]

TARGET_FPS = 10


# =====================================================================
# Video / mask helpers
# =====================================================================

def load_video_frames(video_path, width, height, max_frames=16):
    """Load video → list of PIL images at (width, height)."""
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    src_fps = meta.get("fps", TARGET_FPS)
    step = max(1, round(src_fps / TARGET_FPS))

    frames = []
    for idx, frame in enumerate(reader):
        if idx % step != 0:
            continue
        frame = resize_and_center_crop(frame, width, height)
        frames.append(frame)
        if len(frames) >= max_frames:
            break
    reader.close()
    return [Image.fromarray(f) for f in frames]


def load_video_native(video_path, max_frames=16):
    """Load raw frames at native resolution."""
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    src_fps = meta.get("fps", TARGET_FPS)
    step = max(1, round(src_fps / TARGET_FPS))

    frames = []
    for idx, frame in enumerate(reader):
        if idx % step != 0:
            continue
        frames.append(frame)
        if len(frames) >= max_frames:
            break
    reader.close()
    return frames


def load_masks_as_pil(mask_dir, width, height, num_frames):
    """Load mask PNGs → list of RGB PIL images resized to (width, height)."""
    paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    # Skip non-frame files like mask_vis.mp4 etc
    paths = [p for p in paths if os.path.basename(p)[0].isdigit()]
    masks = []
    for p in paths[:num_frames]:
        m = Image.open(p).convert("RGB").resize((width, height), Image.NEAREST)
        masks.append(m)
    while len(masks) < num_frames:
        masks.append(masks[-1] if masks else Image.new("RGB", (width, height), 0))
    return masks


def load_masks_native(mask_dir, num_frames):
    """Load mask PNGs at native resolution → list of float32 [H,W] arrays."""
    paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    paths = [p for p in paths if os.path.basename(p)[0].isdigit()]
    masks = []
    for p in paths[:num_frames]:
        m = np.array(Image.open(p).convert("L")).astype(np.float32) / 255.0
        masks.append(m)
    while len(masks) < num_frames:
        masks.append(masks[-1] if masks else np.zeros((64, 64), dtype=np.float32))
    return masks


def blend_with_original(orig_frames, relit_frames, masks, width, height):
    """Resize relit to original, paste original foreground back using mask."""
    out = []
    for i, (orig, mask) in enumerate(zip(orig_frames, masks)):
        H, W = orig.shape[:2]
        relit = relit_frames[i] if i < len(relit_frames) else relit_frames[-1]
        if isinstance(relit, Image.Image):
            relit = np.array(relit)
        relit_up = cv2.resize(relit, (W, H), interpolation=cv2.INTER_LANCZOS4)
        alpha = mask[..., None]  # [H,W,1], 1=foreground
        blended = orig.astype(np.float32) * alpha + relit_up.astype(np.float32) * (1 - alpha)
        out.append(np.clip(blended, 0, 255).astype(np.uint8))
    return out


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AnimateDiff sidewalk relighting with background inpainting.",
    )
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Folder with PNG masks from extract_foreground.py (255=foreground).")
    parser.add_argument("--save_dir", type=str, default="output_paint_sidewalk")
    parser.add_argument("--n_lights", type=int, default=5)

    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames for AnimateDiff (default 16).")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="Denoising strength for background inpainting.")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--num_step", type=int, default=50)
    parser.add_argument("--text_guide_scale", type=float, default=2.0)
    parser.add_argument("--negative_prompt", type=str, default="bad quality, worse quality")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sd_model", type=str,
                        default="stablediffusionapi/realistic-vision-v51")
    parser.add_argument("--motion_adapter_model", type=str,
                        default="guoyww/animatediff-motion-adapter-v1-5-3")
    parser.add_argument("--ic_light_model", type=str,
                        default="./models/iclight_sd15_fc.safetensors")
    args = parser.parse_args()

    device = torch.device("cuda")
    adopted_dtype = torch.float16
    set_all_seed(args.seed)

    # ── 1. Load video ─────────────────────────────────────────────────
    print(f"Loading video: {args.video_path}")
    video_list = load_video_frames(args.video_path, args.width, args.height,
                                   max_frames=args.num_frames)
    orig_frames = load_video_native(args.video_path, max_frames=args.num_frames)
    orig_h, orig_w = orig_frames[0].shape[:2]
    num_frames = len(video_list)
    print(f"  {num_frames} frames, model: {args.width}×{args.height}, native: {orig_w}×{orig_h}")

    # ── 2. Load masks ─────────────────────────────────────────────────
    print(f"Loading masks from: {args.mask_dir}")
    mask_list = load_masks_as_pil(args.mask_dir, args.width, args.height, num_frames)
    masks_native = load_masks_native(args.mask_dir, num_frames)
    n_with_fg = sum(1 for m in masks_native if m.max() > 0)
    print(f"  {n_with_fg}/{num_frames} frames have foreground objects")

    # ── 3. Build foreground video (fg pixels only, bg = gray) ─────────
    fg_video_tensor = get_fg_video(
        [np.array(v) for v in video_list],
        [np.array(m) for m in mask_list],
        device, adopted_dtype,
    )

    # ── 4. Load models (identical to lav_paint.py) ────────────────────
    print("Loading models …")

    ## AnimateDiff VDM
    adapter = MotionAdapter.from_pretrained(args.motion_adapter_model)
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        args.sd_model, motion_adapter=adapter,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        args.sd_model, subfolder="scheduler", beta_schedule="linear",
    )
    pipe.enable_vae_slicing()
    pipe = pipe.to(device=device, dtype=adopted_dtype)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    ## IC-Light
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae")
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
    vae = vae.to(device=device, dtype=adopted_dtype)
    unet = unet.to(device=device, dtype=adopted_dtype)
    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

    ## Consistent Light Attention
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

    ic_light_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085, beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True, steps_offset=1,
    )
    ic_light_pipe = StableDiffusionImg2ImgPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=ic_light_scheduler,
        safety_checker=None, requires_safety_checker=False,
        feature_extractor=None, image_encoder=None,
    )
    ic_light_pipe = ic_light_pipe.to(device)

    # ── 5. Random lights & run ────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    chosen_prompts = rng.choice(URBAN_RELIGHTING_PROMPTS, size=args.n_lights, replace=False)
    chosen_bg = rng.choice(BG_SOURCES, size=args.n_lights, p=BG_WEIGHTS)

    os.makedirs(args.save_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(args.video_path))[0]

    print(f"\n{'='*60}")
    print(f"Generating {args.n_lights} × 2 relit videos  ({args.width}×{args.height} → {orig_w}×{orig_h})")
    print(f"{'='*60}")

    for i, (prompt, bg_name) in enumerate(zip(chosen_prompts, chosen_bg)):
        bg_source = BGSource[bg_name]
        relight_prompt = f"sidewalk scene, {prompt}"
        inpaint_prompt = f"urban sidewalk scene, {prompt}"
        prompt_slug = prompt.replace(" ", "_").replace(",", "")[:60]

        print(f"\n[{i+1}/{args.n_lights}] \"{prompt}\"  light: {bg_source.value}")

        generator = torch.manual_seed(args.seed + i)

        with torch.no_grad():
            # Step A: Relight foreground with IC-Light (subtle)
            relighter = Relighter(
                pipeline=ic_light_pipe,
                relight_prompt=relight_prompt,
                bg_source=bg_source,
                generator=generator,
                num_frames=num_frames,
                image_width=args.width,
                image_height=args.height,
            )
            vdm_init_latent = relighter(fg_video_tensor)

            # Step B: AnimateDiff inpainting — high strength on background
            output = pipe(
                ic_light_pipe=ic_light_pipe,
                relight_prompt=relight_prompt,
                bg_source=bg_source,
                mask=mask_list,
                vdm_init_latent=vdm_init_latent,
                video=video_list,
                prompt=inpaint_prompt,
                strength=args.strength,
                negative_prompt=args.negative_prompt,
                guidance_scale=args.text_guide_scale,
                num_inference_steps=args.num_step,
                height=args.height,
                width=args.width,
                generator=generator,
            )

            inpaint_frames = output.frames[0]  # list of PIL or np arrays
            if isinstance(inpaint_frames[0], Image.Image):
                inpaint_np = [np.array(f) for f in inpaint_frames]
            else:
                inpaint_np = list(inpaint_frames)

        # ── Save sample 1: "inpaint" — full pipeline output at original res
        inpaint_upscaled = []
        for f in inpaint_np:
            inpaint_upscaled.append(
                cv2.resize(f, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
            )
        save_name_1 = f"{video_stem}_inpaint_{i:02d}_{prompt_slug}.mp4"
        save_path_1 = os.path.join(args.save_dir, save_name_1)
        imageio.mimwrite(save_path_1, inpaint_upscaled, fps=TARGET_FPS)
        print(f"  [inpaint] {len(inpaint_upscaled)} frames → {save_path_1}")

        # ── Save sample 2: "replace" — inpainted bg + original fg pasted back
        replaced = blend_with_original(
            orig_frames[:num_frames], inpaint_np,
            masks_native[:num_frames], args.width, args.height,
        )
        save_name_2 = f"{video_stem}_replace_{i:02d}_{prompt_slug}.mp4"
        save_path_2 = os.path.join(args.save_dir, save_name_2)
        imageio.mimwrite(save_path_2, replaced, fps=TARGET_FPS)
        print(f"  [replace] {len(replaced)} frames → {save_path_2}")

    print(f"\nDone! {args.n_lights * 2} videos saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
