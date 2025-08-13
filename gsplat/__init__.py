import os
import sys
import torch
import warnings

BACKEND: str = ""

FORCE_BACKEND = os.getenv("GSPLAT_BACKEND", "").lower()

if FORCE_BACKEND == "cuda" or (FORCE_BACKEND == "" and torch.cuda.is_available()):
    try:
        BACKEND = "cuda"
        from .cuda._wrapper import (
            RollingShutterType,
            fully_fused_projection,
            fully_fused_projection_2dgs,
            fully_fused_projection_with_ut,
            isect_offset_encode,
            isect_tiles,
            proj,
            quat_scale_to_covar_preci,
            rasterize_to_indices_in_range,
            rasterize_to_indices_in_range_2dgs,
            rasterize_to_pixels,
            rasterize_to_pixels_2dgs,
            rasterize_to_pixels_eval3d,
            spherical_harmonics,
            world_to_cam,
        )
        print("gsplat: CUDA backend successfully loaded.", file=sys.stderr)
    except ImportError:
        if FORCE_BACKEND == "cuda":
            print("gsplat: Error! GSPLAT_BACKEND=cuda was set but CUDA backend failed to load.", file=sys.stderr)
        pass

if not BACKEND and (FORCE_BACKEND == "sycl" or FORCE_BACKEND == ""):
    try:
        BACKEND = "sycl"
        from .sycl._wrapper import (
            RollingShutterType,
            fully_fused_projection,
            fully_fused_projection_2dgs,
            fully_fused_projection_with_ut,
            isect_offset_encode,
            isect_tiles,
            proj,
            quat_scale_to_covar_preci,
            rasterize_to_indices_in_range,
            rasterize_to_indices_in_range_2dgs,
            rasterize_to_pixels,
            rasterize_to_pixels_2dgs,
            rasterize_to_pixels_eval3d,
            spherical_harmonics,
            world_to_cam,
        )
        print("gsplat: SYCL backend successfully loaded.", file=sys.stderr)
    except ImportError as e:
        if FORCE_BACKEND == "sycl":
            print(f"gsplat: Error! GSPLAT_BACKEND=sycl was set but SYCL backend failed to load: {e}", file=sys.stderr)
        pass

if not BACKEND:
    print(
        "gsplat: Warning! No high-performance backend (CUDA or SYCL) found.",
        file=sys.stderr,
    )


from .compression import PngCompression
from .exporter import export_splats
from .optimizers import SelectiveAdam
from .rendering import (
    rasterization,
    rasterization_2dgs,
    rasterization_2dgs_inria_wrapper,
    rasterization_inria_wrapper,
)
from .strategy import DefaultStrategy, MCMCStrategy, Strategy
from .version import __version__


__all__ = [
    "BACKEND",
    "PngCompression",
    "DefaultStrategy",
    "MCMCStrategy",
    "Strategy",
    "rasterization",
    "rasterization_2dgs",
    "rasterization_inria_wrapper",
    "spherical_harmonics",
    "isect_offset_encode",
    "isect_tiles",
    "proj",
    "fully_fused_projection",
    "quat_scale_to_covar_preci",
    "rasterize_to_pixels",
    "world_to_cam",
    "rasterize_to_indices_in_range",
    "fully_fused_projection_2dgs",
    "rasterize_to_pixels_2dgs",
    "rasterize_to_indices_in_range_2dgs",
    "rasterization_2dgs_inria_wrapper",
    "RollingShutterType",
    "fully_fused_projection_with_ut",
    "rasterize_to_pixels_eval3d",
    "export_splats",
    "__version__",
    "SelectiveAdam",
    # Note: accumulate and accumulate_2dgs are not typically part of the public API
]