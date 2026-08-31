"""MIMIC data augmentation: corrective behavior expansion.

Two augmentation axes, each wrapping an upstream video model kept as an
unmodified submodule:

- :mod:`mimic.appearance_augmentation` varies lighting, weather and time of
  day while holding geometry and the ego trajectory fixed.
- :mod:`mimic.action_augmentation` varies the trajectory itself, generating
  deviate-and-recover maneuvers with matching action labels.
"""

__all__ = ["action_augmentation", "appearance_augmentation"]
