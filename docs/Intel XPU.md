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
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
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

- Finally, build and install the project's Python extension.

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
| 7k steps  | 24.01   | 29.66  | 27.26   | 26.59  | 28.65   | 28.70 | 26.03 |
| 30k steps |         | 31.89  | 29.14   |        | 30.90   | 31.06 |       |


| SSIM      | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  | 0.6808  | 0.9262 | 0.8865  | 0.8370 | 0.9047  | 0.8945| 0.7378|
| 30k steps |         | 0.9446 | 0.9158  |        | 0.9318  | 0.9239|       |


| LPIPS     | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  | 0.2997  | 0.1462 | 0.1929  | 0.1195 | 0.1220  | 0.2136| 0.2339|
| 30k steps |         | 0.1179 | 0.1414  |        | 0.08607 | 0.1520|       |

| Num GSs   | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  | 3.95 M  | 1.19 M | 1.06 M  | 4.20 M | 1.77 M  | 1.14 M| 4.04 M|
| 30k steps |         | 1.28 M | 1.27 M  |        | 1.90 M  | 1.63 M|       |

### 3DGS Training time and memory

| Mip-NeRF 360 scene    | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
|-----------------------|---------|--------|---------|--------|---------|--------|--------|
| 7k steps Mem (GB)     | 5.846   | 2.009  | 1.730   | 6.177  | 2.737   | 1.861  | 5.921  |
| 30k steps Mem (GB)    |         | 2.043  | 1.986   |        | 2.923   | 2.463  |        |
| 7k steps time (s)     | 588.5   | 534.4  | 625.3   | 793.6  | 919.1   | 645.1  | 519.3  |
| 30k steps time (s)    |         | 2612   | 3520    |        | 5027    | 3317   |        |

### 2DGS Reproduced metrics

| PSNR      | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  | 23.59   | 29.73  | 27.25   | 26.31  | 29.02   | 29.56 | 25.69 |
| 30k steps | 25.33   | 32.13  | 28.90   | 27.39  | 31.33   | 31.43 | 26.72 |


| SSIM      | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  | 0.6578  | 0.9277 | 0.8819  | 0.8222 | 0.9021  | 0.9026| 0.7225|
| 30k steps | 0.7570  | 0.9453 | 0.9097  | 0.8567 | 0.9283  | 0.9249| 0.7743|


| LPIPS     | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  | 0.3091  | 0.1441 | 0.1935  | 0.1289 | 0.1231  | 0.1988| 0.2403|
| 30k steps | 0.1745  | 0.1173 | 0.1503  | 0.08466| 0.09162 | 0.1555| 0.1565|

| Num GSs   | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  | 2.52 M  | 0.911 M| 0.695 M | 2.18 M | 0.856 M | 0.839 M| 2.69 M|
| 30k steps | 3.67 M  | 0.929 M| 0.731 M | 2.39 M | 0.870 M | 1.03 M| 3.30 M|

### 2DGS Training time and memory

| Mip-NeRF 360 scene    | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
|-----------------------|---------|--------|---------|--------|---------|--------|--------|
| 7k steps Mem (GB)     | 4.621   | 2.129  | 1.832   | 4.004  | 2.063   | 2.057  | 4.766  |
| 30k steps Mem (GB)    | 6.491   | 2.129  | 1.854   | 4.278  | 2.063   | 2.224  | 5.802  |
| 7k steps time (s)     | 560.0   | 758.2  | 666.3   | 609.3  | 732.8   | 643.8  | 545.6  |
| 30k steps time (s)    | 3483    | 3308   | 2941    | 3101   | 3196    | 2936   | 3117   |