import os
import sys
import torch

BACKEND: str = ""
torch_acc = torch.cpu
_force_backend = os.getenv("GSPLAT_BACKEND", "").lower()

from .cuda._wrapper import (  # Default to CUDA imports, works even if no CUDA is available
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

if _force_backend == "cuda" or (_force_backend == "" and torch.cuda.is_available()):
    BACKEND = "cuda"
    torch_acc = torch.cuda
    print("gsplat: Using CUDA backend.", file=sys.stderr)
    # Functions already imported above

if (
    not BACKEND
    and _force_backend in ("sycl", "xpu")
    or _force_backend == ""
    and torch.xpu.is_available()
):
    from .sycl._wrapper import (  # Overwrite imports for SYCL backend
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

    BACKEND = "sycl"
    torch_acc = torch.xpu
    print("gsplat: Using SYCL XPU backend.", file=sys.stderr)


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
    "torch_acc",
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
