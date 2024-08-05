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



​      