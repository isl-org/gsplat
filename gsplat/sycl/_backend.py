_C = None

try:
    # Try to import the compiled module (via setup.py or pre-built .so)
    from gsplat import gsplat_sycl_kernels as _C
except ImportError: 
    raise ImportError("Unable to find compiled sycl kernels package")