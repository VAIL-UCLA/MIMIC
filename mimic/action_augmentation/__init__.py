"""Action augmentation: corrective behavior expansion.

Builds deviate-and-recover maneuvers on recorded clips — the robot drifts
laterally off the recorded path and rejoins it — and generates both the
re-rendered video and the matching action label.

Backed by TrajectoryCrafter (ICCV 2025) for the view synthesis, included
as a submodule under ``third_party/`` and used unmodified.
"""

from .labels import LabelData, find_sidecar, load_labels, save_labels
from .trajectory import (
    DEFAULT_HORIZON_S,
    DEFAULT_LABEL_TIMES,
    PROFILES,
    apply_lateral_offset,
    build_augmented_label,
    deviate_and_recover,
    lateral_offset_profile,
    waypoints_from_path,
)

__all__ = [
    "DEFAULT_HORIZON_S",
    "DEFAULT_LABEL_TIMES",
    "PROFILES",
    "LabelData",
    "apply_lateral_offset",
    "build_augmented_label",
    "deviate_and_recover",
    "find_sidecar",
    "lateral_offset_profile",
    "load_labels",
    "save_labels",
    "waypoints_from_path",
]
