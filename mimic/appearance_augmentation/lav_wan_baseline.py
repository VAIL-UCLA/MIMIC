"""WAN-based relighting — pure inference, no masks / blending / upscaling.

Uses the same random prompt pool, frame count, and resolution settings as
lav_wan_sidewalk.py, but outputs raw relighted video at model resolution.

Usage:
    python lav_wan_baseline.py --video_path path/to/video.mp4 \
        [--save_dir output_baseline] [--n_lights 5] [--seed 42]
"""

import os
import argparse
from types import MethodType

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

# ── Prompt pool (identical to lav_wan_sidewalk.py) ────────────────────
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
DURATION_SEC = 8
WAN_NUM_FRAMES = TARGET_FPS * DURATION_SEC + 1  # 81
OUTPUT_FRAMES = TARGET_FPS * DURATION_SEC        # 80


def _round16(x: int) -> int:
    return max(16, round(x / 16) * 16)


def compute_model_resolution(
    orig_w: int, orig_h: int, max_side: int = 720,
) -> tuple[int, int]:
    scale = min(max_side / max(orig_w, orig_h), 1.0)
    return _round16(int(orig_w * scale)), _round16(int(orig_h * scale))


def load_video(video_path: str, width: int, height: int, max_frames: int):
    """Load video, resize+crop to (width, height), return PIL list."""
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    src_fps = meta.get("fps", TARGET_FPS)
    step = max(1, round(src_fps / TARGET_FPS))

    frames = []
    for idx, frame in enumerate(reader):
        if idx % step != 0:
            continue
        frame = resize_and_center_crop(frame, width, height)
        frames.append(Image.fromarray(frame))
        if len(frames) >= max_frames:
            break
    reader.close()

    while len(frames) < max_frames:
        frames.append(frames[-1])

    return frames


def main():
    parser = argparse.ArgumentParser(
        description="WAN relighting baseline — pure inference, same prompts/length as sidewalk.",
    )
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="output_baseline")
    parser.add_argument("--n_lights", type=int, default=5)

    parser.add_argument("--max_side", type=int, default=720)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--num_step", type=int, default=15)
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

    # ── 1. Probe video for resolution, then load at model res ─────────
    print(f"Loading video: {args.video_path}")
    reader = imageio.get_reader(args.video_path)
    probe = reader.get_data(0)
    reader.close()
    orig_h, orig_w = probe.shape[:2]

    model_w, model_h = compute_model_resolution(orig_w, orig_h, args.max_side)
    print(f"  Native {orig_w}×{orig_h}  →  Model {model_w}×{model_h}")

    video_list = load_video(args.video_path, model_w, model_h, WAN_NUM_FRAMES)
    print(f"  {len(video_list)} frames loaded")

    # ── 2. Load models ────────────────────────────────────────────────
    print("Loading models …")

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
        vae=vae_sd, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=ic_light_scheduler,
        safety_checker=None, requires_safety_checker=False,
        feature_extractor=None, image_encoder=None,
    )
    ic_light_pipe = ic_light_pipe.to(device=device, dtype=adopted_dtype)
    ic_light_pipe.vae.requires_grad_(False)
    ic_light_pipe.unet.requires_grad_(False)

    # ── 3. Run ────────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    chosen_prompts = rng.choice(URBAN_RELIGHTING_PROMPTS, size=args.n_lights, replace=False)
    chosen_bg = rng.choice(BG_SOURCES, size=args.n_lights, p=BG_WEIGHTS)

    os.makedirs(args.save_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
    generator = torch.manual_seed(args.seed)
    vdm_prompt = "driving through urban sidewalk scenery"
    num_inference_steps = int(round(args.num_step / args.strength))

    print(f"\n{'='*60}")
    print(f"Generating {args.n_lights} relit versions  ({model_w}×{model_h})")
    print(f"{'='*60}")

    for i, (prompt, bg_name) in enumerate(zip(chosen_prompts, chosen_bg)):
        if i < 1:
            continue
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
            frames = (output.frames[0] * 255).astype(np.uint8)

        frames = frames[:OUTPUT_FRAMES]
        prompt_slug = prompt.replace(" ", "_").replace(",", "")[:60]
        save_name = f"{video_stem}_relight_{i:02d}_{prompt_slug}.mp4"
        save_path = os.path.join(args.save_dir, save_name)
        imageio.mimwrite(save_path, frames, fps=TARGET_FPS)
        print(f"  Saved {len(frames)} frames @ {model_w}×{model_h} → {save_path}")

    print(f"\nDone! {args.n_lights} relit videos saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
