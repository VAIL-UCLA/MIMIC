"""Prompt store for appearance augmentation.

Prompts describe a *lighting / weather / time-of-day* condition. The relighting
model re-renders the scene under that condition while the video diffusion model
holds temporal consistency, so the geometry and the ego-trajectory are unchanged
and only the appearance varies.

Prompts are grouped into categories so a run can be restricted to a subset
(e.g. only night and rain conditions):

    from mimic.appearance_augmentation.prompts import build_prompt_pool, sample_prompts

    pool = build_prompt_pool(categories=["night", "rain"])
    picks = sample_prompts(4, seed=42, categories=["night", "rain"])

List them from the shell:

    python -m mimic.appearance_augmentation.prompts --list
    python -m mimic.appearance_augmentation.prompts --categories night rain --sample 5
"""

from __future__ import annotations

import random

# Prepended to every real-world lighting prompt to anchor the scene type.
SCENE_PREFIX = "sidewalk scene, "

# Prompt describing the scene content for the video diffusion model (not the lighting).
VDM_PROMPT = "urban sidewalk navigation, walking along pedestrian path"

NEGATIVE_PROMPT = (
    "bad quality, worse quality, low quality, low resolution, blurry, blur, out of focus, "
    "distorted, deformed, disfigured, ugly, bad anatomy, wrong proportions, "
    "oversaturated, underexposed, overexposed, flat lighting, harsh shadows, "
    "artifacts, noise, grain, watermark, text, logo, duplicate, mutilated, "
    "cropped, jpeg artifacts, compression artifacts, flickering, inconsistent lighting"
)


# =====================================================================
# Real-world lighting conditions, by category
# =====================================================================

URBAN_RELIGHTING_PROMPTS: dict[str, list[str]] = {
    "daytime_sun": [
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
    ],
    "overcast": [
        "overcast morning with diffuse light",
        "overcast afternoon with gray uniform light",
        "partly cloudy sky with scattered sunlight",
        "thick overcast with flat shadowless lighting",
        "thin cloud cover with soft diffuse sunlight",
        "bright overcast day with even illumination everywhere",
        "heavy cloud cover creating low-contrast muted colors",
        "silver overcast sky reflecting off wet surfaces",
    ],
    "seasons": [
        "summer midday with strong ground reflection",
        "winter low-angle sun with long shadows",
        "autumn afternoon with golden light",
        "spring morning with fresh bright light through scattered clouds",
        "winter sunrise with cold low-angle light",
        "late autumn dusk with warm amber tones and bare trees",
        "midsummer evening with extended golden hour",
        "early spring overcast with cool blue-white light",
    ],
    "urban_shadows": [
        "urban canyon shadows between tall buildings",
        "shadows from overhead bridge or overpass",
        "building shadow edge with half sun half shade",
        "narrow alley with only sky-reflected ambient light",
        "deep urban canyon with light only reaching ground at noon",
        "patchy shadows from trees lining the sidewalk",
        "dappled sunlight filtering through overhead tree canopy",
    ],
    "twilight": [
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
    ],
    "night": [
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
    ],
    "rain": [
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
    ],
    "snow": [
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
    ],
    "fog": [
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
    ],
    "dust": [
        "dust storm with orange-tinted reduced visibility",
        "light dust haze with warm tinted sunlight",
        "sandy wind with particles catching sunlight",
    ],
    "covered": [
        "covered walkway with overhead artificial lighting",
        "parking garage interior with fluorescent ceiling lights",
        "shopping arcade with mixed daylight and shop lighting",
        "train station platform under roof with skylight panels",
        "pedestrian underpass with cold fluorescent tubes",
    ],
    "camera": [
        "slightly overexposed bright scene with blown highlights",
        "underexposed dark scene with visible noise",
        "high dynamic range scene with bright sky and dark ground",
        "backlit scene with strong lens flare from sun",
        "low-light scene with visible camera sensor noise",
        "motion blur from fast camera movement in low light",
    ],
    "geographic": [
        "tropical midday with harsh equatorial sun and deep shadows",
        "northern latitude winter with sun barely above horizon",
        "desert environment with intense direct sun and clear sky",
        "Mediterranean afternoon with warm golden light and blue shadows",
        "Pacific Northwest overcast with soft green-tinted light",
        "Southeast Asian monsoon season overcast with warm humid haze",
        "Nordic summer midnight sun with low warm light from north",
        "Australian outback harsh midday sun with red earth reflection",
    ],
    "mixed": [
        "transitional light from sunshine to sudden cloud cover",
        "mixed indoor-outdoor light at building entrance",
        "sunrise with morning fog creating light rays through trees",
        "night rain with neon shop signs reflecting on wet street",
        "snow falling under orange sodium streetlights",
        "sunset through industrial smoke creating dramatic red sky",
        "bright cloudy day with occasional sun breaks through gaps",
        "late evening with last sunlight on building tops and shadows below",
    ],
}


# =====================================================================
# Simulator / rendering-engine style transfer
# =====================================================================
# These already name the scene, so SCENE_PREFIX is not applied to them.

SIMULATOR_STYLE_PROMPTS: list[str] = [
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

SIMULATOR_CATEGORY = "simulator"

#: Every selectable category name.
CATEGORIES: list[str] = list(URBAN_RELIGHTING_PROMPTS) + [SIMULATOR_CATEGORY]


# =====================================================================
# Light direction, sampled alongside the prompt
# =====================================================================
# Names match the relighting model's BGSource enum. Side lighting is weighted
# highest because it produces the most visible appearance change on a sidewalk.

BG_SOURCES: list[str] = ["TOP", "RIGHT", "LEFT", "BOTTOM"]
BG_WEIGHTS: list[float] = [0.2, 0.35, 0.35, 0.1]


# =====================================================================
# Helpers
# =====================================================================


def build_prompt_pool(
    categories: list[str] | None = None,
    include_simulator: bool = True,
    scene_prefix: str = SCENE_PREFIX,
) -> list[str]:
    """Return a flat, deduplicated prompt pool.

    Args:
        categories: Category names to draw from; ``None`` means all of them.
            Passing an explicit list containing ``"simulator"`` includes the
            simulator prompts regardless of ``include_simulator``.
        include_simulator: Whether to append simulator-style prompts when
            ``categories`` is ``None``.
        scene_prefix: Prepended to real-world lighting prompts to anchor the
            scene type. Simulator prompts already name the scene, so they are
            never prefixed.

    Raises:
        ValueError: If a requested category does not exist, or the selection is
            empty.
    """
    if categories is None:
        selected = list(URBAN_RELIGHTING_PROMPTS)
        if include_simulator:
            selected.append(SIMULATOR_CATEGORY)
    else:
        selected = list(categories)

    unknown = [c for c in selected if c not in CATEGORIES]
    if unknown:
        raise ValueError(
            f"Unknown prompt category: {', '.join(unknown)}. Available: {', '.join(CATEGORIES)}"
        )

    pool: list[str] = []
    for category in selected:
        if category == SIMULATOR_CATEGORY:
            pool.extend(SIMULATOR_STYLE_PROMPTS)
        else:
            pool.extend(scene_prefix + p for p in URBAN_RELIGHTING_PROMPTS[category])

    # Deduplicate while preserving order, so a repeated category cannot bias sampling.
    seen: set[str] = set()
    unique = [p for p in pool if not (p in seen or seen.add(p))]
    if not unique:
        raise ValueError("Prompt pool is empty for the requested categories.")
    return unique


def sample_prompts(
    n: int,
    seed: int | None = None,
    categories: list[str] | None = None,
    include_simulator: bool = True,
) -> list[str]:
    """Sample ``n`` distinct prompts. Returns the whole pool if it holds fewer than ``n``."""
    pool = build_prompt_pool(categories, include_simulator)
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def sample_bg_sources(n: int, seed: int | None = None) -> list[str]:
    """Sample ``n`` light directions (with replacement) using BG_WEIGHTS."""
    rng = random.Random(seed)
    return rng.choices(BG_SOURCES, weights=BG_WEIGHTS, k=n)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the appearance-augmentation prompt store.")
    parser.add_argument("--categories", nargs="+", default=None, help=f"Subset of: {', '.join(CATEGORIES)}")
    parser.add_argument("--no_simulator", action="store_true", help="Exclude simulator-style prompts.")
    parser.add_argument("--list", action="store_true", help="Print the whole pool.")
    parser.add_argument("--sample", type=int, default=0, help="Print N sampled prompts.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pool = build_prompt_pool(args.categories, include_simulator=not args.no_simulator)
    print(f"{len(pool)} prompts across {len(args.categories or CATEGORIES)} categories")
    for name, entries in URBAN_RELIGHTING_PROMPTS.items():
        print(f"  {name:<16} {len(entries)}")
    print(f"  {SIMULATOR_CATEGORY:<16} {len(SIMULATOR_STYLE_PROMPTS)}")

    if args.list:
        print()
        for i, prompt in enumerate(pool):
            print(f"  {i:3d}: {prompt}")
    if args.sample:
        print(f"\nSample (seed={args.seed}):")
        for prompt in sample_prompts(
            args.sample, args.seed, args.categories, include_simulator=not args.no_simulator
        ):
            print(f"  - {prompt}")


if __name__ == "__main__":
    _main()
