# GSPLAT on Intel GPUs

``gsplat`` supports creation and rendering on Intel GPUs through SYCL kernel backend. This provides support for both integrated (Alder Lake Arc and onward) as well as discrete GPUs (Arc Alchemist and newer, such as the A770 and B580).

## Supported Features:

- [x] 3DGS fused training
- [x] 3DGS packed representation
- [x] Distributed training (PyTorch 2.8+)
- [x] MCMC strategy (relocation kernel)
- [x] 2DGS fused training
- [ ] 2DGS packed representation
- [ ] 3DGUT kernels (+FTheta cameras)
- [ ] Fused Bilateral grid kernels (from https://github.com/harry7557558/fused-bilagrid)
- [ ] 3DGS Compression (requires PLAS)
- [ ] `rasterize_to_indices_{2,3}dgs` and `rasterize_to_pixels_from_world_3dgs` kernels (only used internally for testing)

The kernels are optimized and use mixed precision (some data is represented as half), so the results differ slightly from the CUDA kernel results.

## Installing (Linux or Windows):

-   **PyTorch XPU:** Install the PyTorch XPU version.

    ```bash
    python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
    ```

-   **Intel oneAPI Toolkit:** Ensure you have the [Intel oneAPI Toolkit installed](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html). This provides the necessary compilers and libraries for SYCL development.

    **Note:** The OneAPI tolkit version must match the version used to build PyTorch XPU. Check the PyTorch XPU OneAPI version with:

        pip show intel-cmplr-lib-ur   # dependency of torch-xpu
        ...
        Version: 2025.0.5
        ...

- Configure your build environment:

    In Linux:

    ```bash
    source /opt/intel/oneapi/setvars.sh
    ```

    Or in Windows:

    ```ps1
    cmd /k "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    powershell
    cmd /k "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
    powershell
    ```

- Finally, build and install the project's Python extension. This step might take some time.

    ```bash
    pip install --no-build-isolation .
    ```

    Alternately, you can build a wheel for distribution with:

    ```bash
    python -m build --no-isolation --wheel
    ```

## Evaluation

We evaluate gsplat-xpu on the Mip-NeRF 360 dataset and measure PSNR, SSIM, LPIPS and the number of Gaussians used. We also measure the memory used and the run time on an Intel Arc B580 GPU.

### 3DGS Reproduced metrics

| PSNR      | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  |         |        |         |        |         |       |       |
| 30k steps |         |        |         |        |         |       |       |


| SSIM      | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  |         |        |         |        |         |       |       |
| 30k steps |         |        |         |        |         |       |       |


| LPIPS     | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  |         |        |         |        |         |       |       |
| 30k steps |         |        |         |        |         |       |       |

| Num GSs   | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  |         |        |         |        |         |       |       |
| 30k steps |         |        |         |        |         |       |       |

### 3DGS Training time and memory

| Mip-NeRF 360 scene    | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
|-----------------------|---------|--------|---------|--------|---------|--------|--------|
| 7k steps Mem (GB)     |         |        |         |        |         |        |        |
| 30k steps Mem (GB)    |         |        |         |        |         |        |        |
| 7k steps time (s)     |         |        |         |        |         |        |        |
| 30k steps time (s)    |         |        |         |        |         |        |        |

### 2DGS Reproduced metrics

| PSNR      | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  |         |        |         |        |         |       |       |
| 30k steps |         |        |         |        |         |       |       |


| SSIM      | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  |         |        |         |        |         |       |       |
| 30k steps |         |        |         |        |         |       |       |


| LPIPS     | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  |         |        |         |        |         |       |       |
| 30k steps |         |        |         |        |         |       |       |

| Num GSs   | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  |         |        |         |        |         |       |       |
| 30k steps |         |        |         |        |         |       |       |

### 2DGS Training time and memory

| Mip-NeRF 360 scene    | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
|-----------------------|---------|--------|---------|--------|---------|--------|--------|
| 7k steps Mem (GB)     |         |        |         |        |         |        |        |
| 30k steps Mem (GB)    |         |        |         |        |         |        |        |
| 7k steps time (s)     |         |        |         |        |         |        |        |
| 30k steps time (s)    |         |        |         |        |         |        |        |