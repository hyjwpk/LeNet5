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
|  卷积 kernel 优化  | 0.0448s |
| 全连接 kernel 优化 | 0.0271s |

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

### 卷积 kernel 优化

```bash
Profiling application: ./infer
Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   57.16%  24.807ms         3  8.2690ms  278.94us  20.922ms  Linear_ReLu(float*, float*, float*, float*, int, int, bool)
                   19.16%  8.3139ms         1  8.3139ms  8.3139ms  8.3139ms  Conv2_ReLu(float*, float*, float*, float*)
                   16.45%  7.1399ms        11  649.08us  1.2800us  7.1028ms  [CUDA memcpy HtoD]
                    4.41%  1.9155ms         1  1.9155ms  1.9155ms  1.9155ms  Conv1_ReLu(float*, float*, float*, float*)
                    2.68%  1.1623ms         2  581.14us  458.21us  704.07us  MaxPool(float*, float*, int, int, int)
                    0.14%  60.416us         1  60.416us  60.416us  60.416us  [CUDA memcpy DtoH]
      API calls:   96.23%  1.57397s        18  87.443ms  2.6000us  1.56926s  cudaMalloc
                    2.29%  37.390ms         7  5.3414ms  403.60us  21.068ms  cudaDeviceSynchronize
                    0.92%  15.029ms        12  1.2524ms  58.200us  7.0049ms  cudaMemcpy
                    0.38%  6.2163ms        18  345.35us  3.9000us  1.8905ms  cudaFree
                    0.18%  2.9293ms         1  2.9293ms  2.9293ms  2.9293ms  cuDeviceGetPCIBusId
                    0.01%  118.10us         7  16.871us  6.6000us  30.300us  cudaLaunchKernel
                    0.00%  21.200us       101     209ns     100ns     900ns  cuDeviceGetAttribute
                    0.00%  2.5000us         7     357ns     200ns     600ns  cudaGetLastError
                    0.00%  1.5000us         2     750ns     200ns  1.3000us  cuDeviceGet
                    0.00%  1.2000us         3     400ns     200ns     700ns  cuDeviceGetCount
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceTotalMem
                    0.00%     400ns         1     400ns     400ns     400ns  cuModuleGetLoadingMode
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

### 全连接 kernel 优化

```bash
Profiling application: ./infer
Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   36.65%  9.3837ms        11  853.06us  1.2800us  9.3464ms  [CUDA memcpy HtoD]
                   33.05%  8.4616ms         1  8.4616ms  8.4616ms  8.4616ms  Conv2_ReLu(float*, float*, float*, float*)
                   15.21%  3.8935ms         1  3.8935ms  3.8935ms  3.8935ms  Linear1_ReLu(float*, float*, float*, float*)
                    6.79%  1.7387ms         1  1.7387ms  1.7387ms  1.7387ms  Conv1_ReLu(float*, float*, float*, float*)
                    4.40%  1.1268ms         2  563.39us  423.06us  703.72us  MaxPool(float*, float*, int, int, int)
                    3.29%  843.02us         1  843.02us  843.02us  843.02us  Linear2_ReLu(float*, float*, float*, float*)
                    0.37%  94.909us         1  94.909us  94.909us  94.909us  Linear3_ReLu(float*, float*, float*, float*)
                    0.24%  61.502us         1  61.502us  61.502us  61.502us  [CUDA memcpy DtoH]
      API calls:   95.89%  1.05407s        18  58.559ms  3.5000us  1.04747s  cudaMalloc
                    1.58%  17.333ms         7  2.4762ms  222.30us  8.5718ms  cudaDeviceSynchronize
                    1.45%  15.937ms        12  1.3281ms  71.700us  9.3504ms  cudaMemcpy
                    0.55%  6.0910ms        18  338.39us  4.5000us  1.6761ms  cudaFree
                    0.51%  5.6447ms         1  5.6447ms  5.6447ms  5.6447ms  cuDeviceGetPCIBusId
                    0.01%  129.10us         7  18.442us  7.2000us  35.600us  cudaLaunchKernel
                    0.00%  26.600us       101     263ns     100ns  1.7000us  cuDeviceGetAttribute
                    0.00%  2.8000us         7     400ns     200ns     800ns  cudaGetLastError
                    0.00%  1.8000us         3     600ns     300ns  1.1000us  cuDeviceGetCount
                    0.00%  1.6000us         2     800ns     200ns  1.4000us  cuDeviceGet
                    0.00%  1.5000us         1  1.5000us  1.5000us  1.5000us  cuDeviceGetName
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceTotalMem
                    0.00%     500ns         1     500ns     500ns     500ns  cuModuleGetLoadingMode
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetUuid
```

