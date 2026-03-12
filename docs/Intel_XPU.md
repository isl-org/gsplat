# GSPLAT on Intel GPUs

`gsplat` supports creation and rendering on Intel GPUs through the SYCL kernel backend. This provides support for both integrated (Alder Lake Arc and onward) and discrete GPUs (Arc Alchemist and newer, such as the A770 and B580).

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
    pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
    ```

-   **Intel oneAPI Toolkit:** Ensure you have the [Intel oneAPI Toolkit installed](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html). This provides the necessary compilers and libraries for SYCL development.

    **Note:** The OneAPI toolkit version must match the version used to build PyTorch XPU. Check the PyTorch XPU OneAPI version with:

    ```bash
    pip show intel-cmplr-lib-ur   # dependency of torch-xpu
    # ...
    # Version: 2025.3.1
    # ...
    ```

- Configure your build environment:

    In Linux:

    ```bash
    source /opt/intel/oneapi/setvars.sh
    ```

    Or in Windows, setup your Visual Studio build environment and then OneAPI build environment. For example:

    ```ps1
    cmd /k "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
    powershell
    $env:DISTUTILS_USE_SDK=1
    ```

- Finally, build and install the project's Python extension.

    ```bash
    pip install --extra-index-url=https://download.pytorch.org/whl/xpu .
    ```

    Alternately, you can build a wheel for distribution with:

    ```bash
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/xpu python -m build --no-isolation --wheel .
    ```

## Evaluation

We evaluate gsplat-xpu on the Mip-NeRF 360 dataset and measure PSNR, SSIM, LPIPS and the number of Gaussians used. We also measure the memory used and the run time on an Intel Arc B580 dGPU and an Intel Arc B390 iGPU. To run the evaluation yourself, download the MIPS-NeRF 360 dataset and install other requirements:

    ```bash
    cd examples
    pip install --extra-index-url=https://download.pytorch.org/whl/xpu -r requirements_xpu.txt
    # download mipnerf_360 benchmark data
    python datasets/download_dataset.py
    ```

The last command will also build and install the `fused-ssim` package. Before running benchmarks, you can add `--max-steps 7000` to each `simple_trainer.py` command in `benchmarks/basic{,_2dgs}.sh`, if you have limited memory, or want to run the training faster. Run the benchmarks with:

    ```bash
    # run batch evaluation
    bash benchmarks/basic.sh
    bash benchmarks/basic_2dgs.sh
    ```

### Arc B580 dGPU

#### 3DGS Reproduced metrics

| PSNR      | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  | 24.01   | 29.66  | 27.26   | 26.59  | 28.65   | 28.70 | 26.03 |
| 30k steps | [^1]    | 31.89  | 29.14   | [^1]   | 30.90   | 31.06 | [^1]  |


| SSIM      | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  | 0.6808  | 0.9262 | 0.8865  | 0.8370 | 0.9047  | 0.8945| 0.7378|
| 30k steps | [^1]    | 0.9446 | 0.9158  | [^1]   | 0.9318  | 0.9239| [^1]  |


| LPIPS     | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  | 0.2997  | 0.1462 | 0.1929  | 0.1195 | 0.1220  | 0.2136| 0.2339|
| 30k steps | [^1]    | 0.1179 | 0.1414  | [^1]   | 0.08607 | 0.1520| [^1]  |

| Num GSs   | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| 7k steps  | 3.95 M  | 1.19 M | 1.06 M  | 4.20 M | 1.77 M  | 1.14 M| 4.04 M|
| 30k steps | [^1]    | 1.28 M | 1.27 M  | [^1]   | 1.90 M  | 1.63 M| [^1]  |

#### 3DGS Training time and memory

| Mip-NeRF 360 scene    | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
|-----------------------|---------|--------|---------|--------|---------|--------|--------|
| 7k steps Mem (GB)     | 5.846   | 2.009  | 1.730   | 6.177  | 2.737   | 1.861  | 5.921  |
| 30k steps Mem (GB)    | [^1]    | 2.043  | 1.986   | [^1]   | 2.923   | 2.463  | [^1]   |
| 7k steps time (s)     | 588.5   | 534.4  | 625.3   | 793.6  | 919.1   | 645.1  | 519.3  |
| 30k steps time (s)    | [^1]    | 2612   | 3520    | [^1]   | 5027    | 3317   | [^1]   |

#### 2DGS Reproduced metrics

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

#### 2DGS Training time and memory

| Mip-NeRF 360 scene    | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
|-----------------------|---------|--------|---------|--------|---------|--------|--------|
| 7k steps Mem (GB)     | 4.621   | 2.129  | 1.832   | 4.004  | 2.063   | 2.057  | 4.766  |
| 30k steps Mem (GB)    | 6.491   | 2.129  | 1.854   | 4.278  | 2.063   | 2.224  | 5.802  |
| 7k steps time (s)     | 560.0   | 758.2  | 666.3   | 609.3  | 732.8   | 643.8  | 545.6  |
| 30k steps time (s)    | 3483    | 3308   | 2941    | 3101   | 3196    | 2936   | 3117   |

[^1]: Out of memory.

### Arc B390 iGPU

#### 3DGS Reproduced metrics

| 7k steps  | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| PSNR      | 24.02   | 29.72  | 27.40   | 26.61  | 29.24   | 29.56 | 25.31 |
| SSIM      | 0.6513  | 0.9252 | 0.8914  | 0.8369 | 0.9173  | 0.9039| 0.6955|
| LPIPS     | 0.3558  | 0.1525 | 0.1891  | 0.1198 | 0.1102  | 0.2037| 0.2941|
| Num GSs   | 3.28 M  | 0.99 M | 0.74 M  | 4.17 M | 1.07 M  | 0.80 M| 4.01 M|

#### 3DGS Training time and memory

| Mip-NeRF 360 scene | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
|--------------------|---------|--------|---------|--------|---------|--------|--------|
| 7k steps Mem (GB)  | 5.016   | 1.642  | 1.264   | 6.142  | 1.678   | 1.366  | 5.932  |
| 7k steps time (s)  | 1346.5  | 1124.1 | 1231.0  | 1958.6 | 1496.2  | 983.1  | 1215.6 |

#### 2DGS Reproduced metrics

| 7k steps  | Bicycle | Bonsai | Counter | Garden | Kitchen | Room  | Stump |
|-----------|---------|--------|---------|--------|---------|-------|-------|
| PSNR      | 23.92   | 29.88  | 27.38   | 25.95  | 29.37   | 29.98 | 25.13 |
| SSIM      | 0.6418  | 0.9301 | 0.8897  | 0.7992 | 0.9128  | 0.9086| 0.6855|
| LPIPS     | 0.3443  | 0.1446 | 0.1843  | 0.1520 | 0.1123  | 0.1919| 0.2902|
| Num GSs   | 2.13 M  | 0.79 M | 0.56 M  | 1.66 M | 0.72 M  | 0.62 M| 2.40 M|

#### 2DGS Training time and memory

| Mip-NeRF 360 scene    | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  |
|-----------------------|---------|--------|---------|--------|---------|--------|--------|
| 7k steps Mem (GB)     | 3.026   | 1.9185 | 1.5837  | 3.026  | 1.8087  | 1.6828 | 4.263  |
| 7k steps time (s)     | 1502.5  | 1868.3 | 2121.1  | 1502.5 | 1816.6  | 1591.3 | 1409.4 |