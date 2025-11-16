import glob
import os
import os.path as osp
import pathlib
import platform
import sys
from setuptools import find_packages, setup
import subprocess as sp

__version__ = None
exec(open("gsplat/version.py", "r").read())

URL = "https://github.com/nerfstudio-project/gsplat"

has_cuda = False
try:
    import torch

    has_cuda = torch.cuda.is_available()
except ImportError:
    pass

has_xpu = False
if not has_cuda:
    try:
        import torch

        has_xpu = torch.xpu.is_available()
    except (ImportError, AttributeError):
        pass

BUILD_SYCL = has_xpu
BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"
MAX_JOBS = os.getenv("MAX_JOBS")
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"
    print(f"Setting MAX_JOBS to {os.environ['MAX_JOBS']}")


from torch.utils.cpp_extension import BuildExtension


class SyclBuildExtension(BuildExtension):
    """
    Custom build class to orchestrate a CMake build for the SYCL backend.
    """

    def run(self):
        print("--- Running SYCL build via CMake ---")
        sycl_dir = os.path.abspath("gsplat/sycl")
        build_dir = os.path.join(self.build_temp, "sycl")
        os.makedirs(build_dir, exist_ok=True)
        jobs = os.getenv("MAX_JOBS", "10")

        install_dir = os.path.abspath(self.build_lib)

        sp.check_call(
            [
                "cmake",
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.join(install_dir, 'gsplat')}",
                sycl_dir,
            ],
            cwd=build_dir,
        )
        sp.check_call(
            ["cmake", "--build", ".", "--config", "Release", "--", "-v", f"-j{jobs}"],
            cwd=build_dir,
        )


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    import torch
    from torch.__config__ import parallel_info
    from torch.utils.cpp_extension import CUDAExtension

    extensions_dir = osp.join("gsplat", "cuda")
    sources = glob.glob(osp.join(extensions_dir, "csrc", "*.cu")) + glob.glob(
        osp.join(extensions_dir, "csrc", "*.cpp")
    )
    sources += [osp.join(extensions_dir, "ext.cpp")]

    undef_macros = []
    define_macros = []

    extra_compile_args = {"cxx": ["-O3"]}
    if not os.name == "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
    nvcc_flags += ["-O3", "--use_fast_math", "-std=c++17"]
    if LINE_INFO:
        nvcc_flags += ["-lineinfo"]
    if torch.version.hip:
        # USE_ROCM was added to later versions of PyTorch.
        # Define here to support older PyTorch versions as well:
        define_macros += [("USE_ROCM", None)]
        undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
    else:
        nvcc_flags += ["--expt-relaxed-constexpr"]

    # GLM/Torch has spammy and very annoyingly verbose warnings that this suppresses
    nvcc_flags += ["-diag-suppress", "20012,186"]
    extra_compile_args["nvcc"] = nvcc_flags
    if sys.platform == "win32":
        extra_compile_args["nvcc"] += [
            "-DWIN32_LEAN_AND_MEAN",
            "-allow-unsupported-compiler",
        ]

    current_dir = pathlib.Path(__file__).parent.resolve()
    glm_path = osp.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm")
    include_dirs = [glm_path, osp.join(current_dir, "gsplat", "cuda", "include")]

    extension = CUDAExtension(
        "gsplat.csrc",
        sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    return [extension]


ext_modules = []
cmdclass = {}
packages_to_find = find_packages()
from setuptools import Extension

if BUILD_SYCL:
    print("--- Configuring for SYCL build ---")
    cmdclass = {"build_ext": SyclBuildExtension}
    ext_modules.append(Extension("gsplat.gsplat_sycl_kernels", sources=[]))
elif not BUILD_NO_CUDA:
    print("--- Configuring for CUDA build ---")
    ext_modules = get_extensions()
    cmdclass = {"build_ext": get_ext()}
else:
    print("--- Building without any C++/CUDA/SYCL extensions ---")

setup(
    name="gsplat",
    version=__version__,
    description="Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda, sycl",
    url=URL,
    download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",
    python_requires=">=3.8",  # Updated to match your CMake
    install_requires=[
        "ninja",
        "numpy",
        "jaxtyping",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    extras_require={
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml==6.0",
            "build",
            "twine",
        ],
        "sycl": ["pybind11>=2.10"],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=packages_to_find,
    include_package_data=True,
    zip_safe=False,
)

if need_to_unset_max_jobs:
    print("Unsetting MAX_JOBS")
    os.environ.pop("MAX_JOBS")
