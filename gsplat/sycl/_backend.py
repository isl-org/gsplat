_C = None

import os

import gsplat

if os.name == "nt":
    import torch
    from sysconfig import get_path

    dllpath = [
        os.add_dll_directory(torch.__path__[0] + "/lib"),  # for torch libs
        os.add_dll_directory(get_path("data") + "/Library/bin"),  # for sycl libs
    ]

try:
    # Try to import the compiled module (via setup.py or pre-built .so)
    from gsplat import gsplat_sycl_kernels as _C
except ImportError:
    raise ImportError("Unable to find compiled sycl kernels package")

if os.name == "nt":
    for dp in dllpath:
        dp.close()

__all__ = ["_C"]
