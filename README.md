# LeNet5 推理 CUDA 优化

## 硬件信息

CPU  	Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz

GPU   	NVIDIA GeForce RTX 2060

## 性能数据（推理 1w 张图片）

|                    |  时间   |
| :----------------: | :-----: |
| torch cpu baseline | 0.3000s |
| torch gpu baseline | 0.0150s |
|   cuda baseline    | 6.3073s |
|    图片并行推理    | 0.0527s |

## prof 结果

### cuda baseline

```bash
Profiling application: ./infer
Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   52.43%  352.50ms     30000  11.749us  7.5520us  28.448us  Linear_ReLu(float*, float*, float*, float*, int, bool)
                   33.64%  226.20ms     20000  11.310us  5.2800us  32.097us  Conv_ReLu(float*, float*, float*, float*, int, int, int, int)
                   10.74%  72.233ms     20000  3.6110us  2.6240us  15.072us  MaxPool(float*, float*, int, int)
                    2.04%  13.746ms     10010  1.3730us  1.0880us  20.448us  [CUDA memcpy HtoD]
                    1.14%  7.6724ms     10000     767ns     639ns  4.0320us  [CUDA memcpy DtoH]
      API calls:   66.33%  5.07250s     70001  72.463us  1.1000us  4.5818ms  cudaDeviceSynchronize
                   18.41%  1.40806s        18  78.225ms  1.7000us  1.40800s  cudaMalloc
                   10.22%  781.43ms     20010  39.052us  2.8000us  2.3014ms  cudaMemcpy
                    4.78%  365.79ms     70000  5.2250us  3.0000us  838.80us  cudaLaunchKernel
                    0.19%  14.672ms     70000     209ns     100ns  363.10us  cudaGetLastError
                    0.05%  3.8270ms         1  3.8270ms  3.8270ms  3.8270ms  cuDeviceGetPCIBusId
                    0.01%  542.60us        18  30.144us  1.5000us  439.20us  cudaFree
                    0.00%  22.100us       101     218ns     100ns  1.2000us  cuDeviceGetAttribute
                    0.00%  1.8000us         3     600ns     200ns  1.3000us  cuDeviceGetCount
                    0.00%  1.6000us         2     800ns     300ns  1.3000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     600ns         1     600ns     600ns     600ns  cuModuleGetLoadingMode
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

### 图片并行推理

```bash
Profiling application: ./infer
Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   46.80%  24.829ms         3  8.2763ms  278.56us  20.942ms  Linear_ReLu(float*, float*, float*, float*, int, int, bool)
                   35.33%  18.744ms         2  9.3718ms  8.6073ms  10.136ms  Conv_ReLu(float*, float*, float*, float*, int, int, int, int, int)
                   15.58%  8.2652ms        11  751.38us  1.3120us  8.2264ms  [CUDA memcpy HtoD]
                    2.19%  1.1606ms         2  580.31us  456.32us  704.29us  MaxPool(float*, float*, int, int, int)
                    0.11%  60.673us         1  60.673us  60.673us  60.673us  [CUDA memcpy DtoH]
      API calls:   94.96%  1.41670s        18  78.706ms  3.3000us  1.40981s  cudaMalloc
                    3.09%  46.046ms         7  6.5780ms  432.90us  21.101ms  cudaDeviceSynchronize
                    0.99%  14.792ms        12  1.2327ms  62.100us  8.0592ms  cudaMemcpy
                    0.66%  9.9202ms        18  551.12us  3.3000us  3.7833ms  cudaFree
                    0.29%  4.2612ms         1  4.2612ms  4.2612ms  4.2612ms  cuDeviceGetPCIBusId
                    0.01%  166.60us         7  23.800us  8.7000us  43.600us  cudaLaunchKernel
                    0.00%  24.900us       101     246ns     100ns  1.5000us  cuDeviceGetAttribute
                    0.00%  4.2000us         7     600ns     200ns  1.1000us  cudaGetLastError
                    0.00%  2.1000us         3     700ns     200ns  1.1000us  cuDeviceGetCount
                    0.00%  1.6000us         2     800ns     200ns  1.4000us  cuDeviceGet
                    0.00%  1.4000us         1  1.4000us  1.4000us  1.4000us  cuDeviceGetName
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceTotalMem
                    0.00%     600ns         1     600ns     600ns     600ns  cuModuleGetLoadingMode
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid
```

