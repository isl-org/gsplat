"""Tests for the backend functions.

Usage:

pytest <THIS_PY_FILE> -s

# To force a specific backend for testing:
"""

import math
import os

import pytest
import torch
from typing_extensions import Literal, Tuple, assert_never

# Import the gsplat library, which will run the __init__.py and select a backend.
import gsplat
from gsplat._helper import load_test_data

if gsplat.BACKEND == "sycl":
    device = torch.device("xpu:0")
elif gsplat.BACKEND == "cuda":
    device = torch.device("cuda:0")
else:
    device = None

requires_backend = pytest.mark.skipif(
    gsplat.BACKEND not in ("cuda", "sycl"),
    reason="No CUDA or SYCL XPU backend available",
)
requires_cuda = pytest.mark.skipif(
    gsplat.BACKEND != "cuda", reason="Test requires CUDA backend"
)


def expand(data: dict, batch_dims: Tuple[int, ...]):
    """Helper function to expand test data with batch dimensions."""
    ret = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and len(batch_dims) > 0:
            new_shape = batch_dims + v.shape
            ret[k] = v.expand(new_shape)
        else:
            ret[k] = v
    return ret


@pytest.fixture
def test_data():
    """Loads test data and moves it to the active device."""
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }


@requires_backend
@pytest.mark.parametrize("triu", [False, True])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_quat_scale_to_covar_preci(test_data, triu: bool, batch_dims: Tuple[int, ...]):

    from gsplat._torch_impl import _quat_scale_to_covar_preci

    torch.manual_seed(42)
    test_data = expand(test_data, batch_dims)
    quats = test_data["quats"]
    scales = test_data["scales"]
    quats.requires_grad = True
    scales.requires_grad = True

    covars, precis = gsplat.quat_scale_to_covar_preci(quats, scales, triu=triu)
    _covars, _precis = _quat_scale_to_covar_preci(quats, scales, triu=triu)
    torch.testing.assert_close(covars, _covars)

    v_covars = torch.randn_like(covars)
    v_precis = torch.randn_like(precis) * 0.01
    v_quats, v_scales = torch.autograd.grad(
        (covars * v_covars + precis * v_precis).sum(), (quats, scales)
    )
    _v_quats, _v_scales = torch.autograd.grad(
        (_covars * v_covars + _precis * v_precis).sum(), (quats, scales)
    )
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e0, atol=1e-1)
    torch.testing.assert_close(v_scales, _v_scales, rtol=1e0, atol=1e-1)


@requires_backend
@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_proj(test_data, camera_model: str, batch_dims: Tuple[int, ...]):

    from gsplat._torch_impl import (
        _fisheye_proj,
        _ortho_proj,
        _persp_proj,
        _world_to_cam,
    )

    torch.manual_seed(42)
    test_data = expand(test_data, batch_dims)
    Ks, viewmats, height, width = (
        test_data["Ks"],
        test_data["viewmats"],
        test_data["height"],
        test_data["width"],
    )

    covars, _ = gsplat.quat_scale_to_covar_preci(
        test_data["quats"], test_data["scales"]
    )
    means, covars = _world_to_cam(test_data["means"], covars, viewmats)
    means.requires_grad = True
    covars.requires_grad = True

    means2d, covars2d = gsplat.proj(means, covars, Ks, width, height, camera_model)
    if camera_model == "ortho":
        _means2d, _covars2d = _ortho_proj(means, covars, Ks, width, height)
    elif camera_model == "fisheye":
        _means2d, _covars2d = _fisheye_proj(means, covars, Ks, width, height)
    elif camera_model == "pinhole":
        _means2d, _covars2d = _persp_proj(means, covars, Ks, width, height)
    else:
        assert_never(camera_model)

    torch.testing.assert_close(means2d, _means2d, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(covars2d, _covars2d, rtol=1e-1, atol=3e-2)

    v_means2d, v_covars2d = torch.randn_like(means2d), torch.randn_like(covars2d)
    v_means, v_covars = torch.autograd.grad(
        (means2d * v_means2d).sum() + (covars2d * v_covars2d).sum(), (means, covars)
    )
    _v_means, _v_covars = torch.autograd.grad(
        (_means2d * v_means2d).sum() + (_covars2d * v_covars2d).sum(), (means, covars)
    )
    torch.testing.assert_close(v_means, _v_means, rtol=6e-1, atol=1e-2)
    torch.testing.assert_close(v_covars, _v_covars, rtol=1e-1, atol=1e-1)


@requires_backend
@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("calc_compensations", [True, False])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_projection(
    test_data,
    fused: bool,
    calc_compensations: bool,
    camera_model: str,
    batch_dims: Tuple[int, ...],
):

    from gsplat._torch_impl import _fully_fused_projection

    torch.manual_seed(42)
    test_data = expand(test_data, batch_dims)
    Ks, viewmats, height, width = (
        test_data["Ks"],
        test_data["viewmats"],
        test_data["height"],
        test_data["width"],
    )
    quats, scales, means = test_data["quats"], test_data["scales"], test_data["means"]
    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    if fused:
        radii, means2d, depths, conics, compensations = gsplat.fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
    else:
        covars, _ = gsplat.quat_scale_to_covar_preci(quats, scales, triu=True)
        radii, means2d, depths, conics, compensations = gsplat.fully_fused_projection(
            means,
            covars,
            None,
            None,
            viewmats,
            Ks,
            width,
            height,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )

    _covars, _ = gsplat.quat_scale_to_covar_preci(quats, scales, triu=False)
    _radii, _means2d, _depths, _conics, _compensations = _fully_fused_projection(
        means,
        _covars,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=calc_compensations,
        camera_model=camera_model,
    )

    valid = (radii > 0).all(dim=-1) & (_radii > 0).all(dim=-1)
    torch.testing.assert_close(radii, _radii, rtol=0, atol=1)
    torch.testing.assert_close(means2d[valid], _means2d[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(depths[valid], _depths[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(conics[valid], _conics[valid], rtol=1e-4, atol=1e-4)
    if calc_compensations:
        torch.testing.assert_close(
            compensations[valid], _compensations[valid], rtol=1e-4, atol=1e-3
        )

    v_means2d, v_depths, v_conics = (
        torch.randn_like(means2d) * valid[..., None],
        torch.randn_like(depths) * valid,
        torch.randn_like(conics) * valid[..., None],
    )
    v_compensations = (
        torch.randn_like(compensations) * valid if calc_compensations else 0
    )
    grad_sum = (
        (means2d * v_means2d).sum()
        + (depths * v_depths).sum()
        + (conics * v_conics).sum()
        + ((compensations * v_compensations).sum() if calc_compensations else 0)
    )
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        grad_sum, (viewmats, quats, scales, means)
    )

    _grad_sum = (
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum()
        + ((_compensations * v_compensations).sum() if calc_compensations else 0)
    )
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        _grad_sum, (viewmats, quats, scales, means)
    )

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(v_quats, _v_quats, rtol=2e-1, atol=2e-2)
    torch.testing.assert_close(v_scales, _v_scales, rtol=5e-1, atol=2e-1)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=6e-2)


@requires_backend
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("sparse_grad", [False])
@pytest.mark.parametrize("calc_compensations", [False, True])
@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_fully_fused_projection_packed(
    test_data,
    fused: bool,
    sparse_grad: bool,
    calc_compensations: bool,
    camera_model: str,
    batch_dims: Tuple[int, ...],
):

    torch.manual_seed(42)
    test_data = expand(test_data, batch_dims)
    Ks, viewmats, height, width = (
        test_data["Ks"],
        test_data["viewmats"],
        test_data["height"],
        test_data["width"],
    )
    quats, scales, means = test_data["quats"], test_data["scales"], test_data["means"]
    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    if fused:
        res = gsplat.fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            packed=True,
            sparse_grad=sparse_grad,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
        (
            _radii,
            _means2d,
            _depths,
            _conics,
            _compensations,
        ) = gsplat.fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            packed=False,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
    else:
        covars, _ = gsplat.quat_scale_to_covar_preci(quats, scales, triu=True)
        res = gsplat.fully_fused_projection(
            means,
            covars,
            None,
            None,
            viewmats,
            Ks,
            width,
            height,
            packed=True,
            sparse_grad=sparse_grad,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
        (
            _radii,
            _means2d,
            _depths,
            _conics,
            _compensations,
        ) = gsplat.fully_fused_projection(
            means,
            covars,
            None,
            None,
            viewmats,
            Ks,
            width,
            height,
            packed=False,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )

    (
        batch_ids,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        compensations,
    ) = res
    B, C, N = math.prod(batch_dims), viewmats.shape[-3], means.shape[-2]

    # Unpack for comparison
    sparse_shape = (B, C, N)
    indices = torch.stack([batch_ids, camera_ids, gaussian_ids])
    __radii = (
        torch.sparse_coo_tensor(indices, radii, sparse_shape + (2,))
        .to_dense()
        .reshape(batch_dims + (C, N, 2))
    )
    __means2d = (
        torch.sparse_coo_tensor(indices, means2d, sparse_shape + (2,))
        .to_dense()
        .reshape(batch_dims + (C, N, 2))
    )
    __depths = (
        torch.sparse_coo_tensor(indices, depths, sparse_shape)
        .to_dense()
        .reshape(batch_dims + (C, N))
    )
    __conics = (
        torch.sparse_coo_tensor(indices, conics, sparse_shape + (3,))
        .to_dense()
        .reshape(batch_dims + (C, N, 3))
    )
    if calc_compensations:
        __compensations = (
            torch.sparse_coo_tensor(indices, compensations, sparse_shape)
            .to_dense()
            .reshape(batch_dims + (C, N))
        )

    sel = (__radii > 0).all(dim=-1) & (_radii > 0).all(dim=-1)
    torch.testing.assert_close(__radii[sel], _radii[sel], rtol=0, atol=1)
    torch.testing.assert_close(__means2d[sel], _means2d[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(__depths[sel], _depths[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(__conics[sel], _conics[sel], rtol=1e-4, atol=1e-4)
    if calc_compensations:
        torch.testing.assert_close(
            __compensations[sel], _compensations[sel], rtol=1e-4, atol=1e-3
        )

    v_means2d, v_depths, v_conics = (
        torch.randn_like(_means2d) * sel[..., None],
        torch.randn_like(_depths) * sel,
        torch.randn_like(_conics) * sel[..., None],
    )
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d[sel]).sum()
        + (depths * v_depths[sel]).sum()
        + (conics * v_conics[sel]).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )
    if sparse_grad:
        v_quats, v_scales, v_means = (
            v_quats.to_dense(),
            v_scales.to_dense(),
            v_means.to_dense(),
        )

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_scales, _v_scales, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-3, atol=1e-3)


@requires_backend
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_isect(test_data, batch_dims: Tuple[int, ...]):

    from gsplat._torch_impl import _isect_offset_encode, _isect_tiles

    torch.manual_seed(42)
    B, C, N = math.prod(batch_dims), 3, 1000
    I, width, height = B * C, 40, 60

    test_data = {
        "means2d": torch.randn(C, N, 2, device=device) * width,
        "radii": torch.randint(0, width, (C, N, 2), device=device, dtype=torch.int32),
        "depths": torch.rand(C, N, device=device),
    }
    test_data = expand(test_data, batch_dims)
    means2d, radii, depths = (
        test_data["means2d"],
        test_data["radii"],
        test_data["depths"],
    )

    tile_size = 16
    tile_width, tile_height = math.ceil(width / tile_size), math.ceil(
        height / tile_size
    )

    tiles_per_gauss, isect_ids, flatten_ids = gsplat.isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = gsplat.isect_offset_encode(isect_ids, I, tile_width, tile_height)

    _tiles_per_gauss, _isect_ids, _gauss_ids = _isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    _isect_offsets = _isect_offset_encode(_isect_ids, I, tile_width, tile_height)

    torch.testing.assert_close(tiles_per_gauss, _tiles_per_gauss)
    torch.testing.assert_close(isect_ids, _isect_ids)
    torch.testing.assert_close(flatten_ids, _gauss_ids)
    torch.testing.assert_close(isect_offsets, _isect_offsets)


@requires_backend
@pytest.mark.parametrize("channels", [3, 32, 128])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_rasterize_to_pixels(test_data, channels: int, batch_dims: Tuple[int, ...]):

    from gsplat._torch_impl import _rasterize_to_pixels

    torch.manual_seed(42)
    N, C = test_data["means"].shape[-2], test_data["viewmats"].shape[-3]
    I = math.prod(batch_dims) * C
    test_data.update(
        {
            "colors": torch.rand(C, N, channels, device=device),
            "backgrounds": torch.rand((C, channels), device=device),
        }
    )
    test_data = expand(test_data, batch_dims)
    Ks, viewmats, height, width = (
        test_data["Ks"],
        test_data["viewmats"],
        test_data["height"],
        test_data["width"],
    )
    quats, scales, means, opacities = (
        test_data["quats"],
        test_data["scales"] * 0.1,
        test_data["means"],
        test_data["opacities"],
    )
    colors, backgrounds = test_data["colors"], test_data["backgrounds"]

    covars, _ = gsplat.quat_scale_to_covar_preci(
        quats, scales, compute_preci=False, triu=True
    )
    radii, means2d, depths, conics, _ = gsplat.fully_fused_projection(
        means, covars, None, None, viewmats, Ks, width, height
    )
    opacities = torch.broadcast_to(opacities[..., None, :], batch_dims + (C, N))

    tile_size = 16 if channels <= 32 else 4
    tile_width, tile_height = math.ceil(width / float(tile_size)), math.ceil(
        height / float(tile_size)
    )
    tiles_per_gauss, isect_ids, flatten_ids = gsplat.isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = gsplat.isect_offset_encode(
        isect_ids, I, tile_width, tile_height
    ).reshape(batch_dims + (C, tile_height, tile_width))

    means2d.requires_grad = True
    conics.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True
    backgrounds.requires_grad = True

    render_colors, render_alphas = gsplat.rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )

    if gsplat.BACKEND != "sycl":  # nerfacc required for comparison
        _render_colors, _render_alphas = _rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
        )
        torch.testing.assert_close(render_colors, _render_colors)
        torch.testing.assert_close(render_alphas, _render_alphas)

    v_render_colors, v_render_alphas = torch.randn_like(
        render_colors
    ), torch.randn_like(render_alphas)
    grads = torch.autograd.grad(
        (render_colors * v_render_colors).sum()
        + (render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, backgrounds),
    )

    if gsplat.BACKEND != "sycl":  # nerfacc required for comparison
        _grads = torch.autograd.grad(
            (_render_colors * v_render_colors).sum()
            + (_render_alphas * v_render_alphas).sum(),
            (means2d, conics, colors, opacities, backgrounds),
        )
        torch.testing.assert_close(grads[0], _grads[0], rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(grads[1], _grads[1], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(grads[2], _grads[2], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(grads[3], _grads[3], rtol=8e-3, atol=6e-3)
        torch.testing.assert_close(grads[4], _grads[4], rtol=1e-3, atol=1e-3)
