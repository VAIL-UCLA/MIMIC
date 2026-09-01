"""Compatibility shims for the pinned upstream submodules.

TrajectoryCrafter is vendored at a fixed commit and used unmodified, so it is
frozen against the diffusers API of its day. Where a later diffusers removes a
path it depends on, the shim lives here in MIMIC — and has to be provably the
same computation, not merely a way to stop the exception.
"""

import numpy as np
import pytest

tc = pytest.importorskip("mimic.action_augmentation.augment_action")


def _upstream_module():
    """The submodule's transformer namespace, or a skip if it isn't set up."""
    import os
    import sys
    from pathlib import Path

    root = (
        Path(__file__).resolve().parents[1]
        / "mimic/action_augmentation/third_party/TrajectoryCrafter"
    )
    if not (root / "models" / "crosstransformer3d.py").is_file():
        pytest.skip("TrajectoryCrafter submodule not initialized")
    pytest.importorskip("diffusers")

    cwd = Path.cwd()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    try:
        import models.crosstransformer3d as upstream
    except ImportError as exc:
        pytest.skip(f"upstream import failed: {exc}")
    finally:
        os.chdir(cwd)
    return upstream


@pytest.mark.parametrize(
    ("embed_dim", "spatial", "temporal"),
    [(64, (4, 6), 5), (128, (15, 20), 13), (1920, (30, 45), 13)],
)
def test_sincos_shim_matches_the_numpy_path(embed_dim, spatial, temporal):
    from diffusers.models import embeddings as E

    upstream = _upstream_module()
    tc._patch_sincos_for_diffusers()

    # The numpy path raises on diffusers >= 0.40; silence only the version
    # guard so the two implementations can be compared directly.
    original_deprecate = E.deprecate
    E.deprecate = lambda *a, **k: None
    try:
        reference = E._get_3d_sincos_pos_embed_np(
            embed_dim=embed_dim, spatial_size=spatial, temporal_size=temporal
        )
    finally:
        E.deprecate = original_deprecate

    got = upstream.get_3d_sincos_pos_embed(embed_dim, spatial, temporal, 1.0, 1.0)

    assert isinstance(got, np.ndarray), "upstream calls torch.from_numpy on the result"
    assert got.shape == reference.shape
    assert np.allclose(got, reference, atol=1e-6)


def test_patch_is_idempotent():
    upstream = _upstream_module()
    tc._patch_sincos_for_diffusers()
    first = upstream.get_3d_sincos_pos_embed
    tc._patch_sincos_for_diffusers()
    assert upstream.get_3d_sincos_pos_embed is first
