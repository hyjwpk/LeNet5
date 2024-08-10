# LeNet5 推理 CUDA 优化

## 硬件信息

CPU  	Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz

GPU   	NVIDIA GeForce RTX 2060

## 性能数据（推理 1w 张图片）

|                              |  时间   |
| :--------------------------: | :-----: |
|      torch cpu baseline      | 0.3000s |
|      torch gpu baseline      | 0.0150s |
|        cuda baseline         | 6.3073s |
|         图片并行推理         | 0.0527s |
|       卷积 kernel 优化       | 0.0448s |
|      全连接 kernel 优化      | 0.0271s |
|         移除冗余同步         | 0.0240s |
|         使用锁页内存         | 0.0207s |
|    全连接 kernel 访存合并    | 0.0188s |
| 计算访存重叠 + kernel fusion | 0.0137s |

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

### 移除冗余同步

```bash
==58233== Profiling application: ./infer
==58233== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   35.44%  8.3601ms         1  8.3601ms  8.3601ms  8.3601ms  Conv2_ReLu(float*, float*, float*, float*)
                   34.48%  8.1332ms        11  739.38us  1.4400us  8.0940ms  [CUDA memcpy HtoD]
                   13.37%  3.1534ms         1  3.1534ms  3.1534ms  3.1534ms  Linear1_ReLu(float*, float*, float*, float*)
                    8.13%  1.9191ms         1  1.9191ms  1.9191ms  1.9191ms  Conv1_ReLu(float*, float*, float*, float*)
                    3.94%  928.45us         1  928.45us  928.45us  928.45us  Linear2_ReLu(float*, float*, float*, float*)
                    3.04%  717.48us         1  717.48us  717.48us  717.48us  MaxPool1(float*, float*)
                    0.91%  214.62us         1  214.62us  214.62us  214.62us  MaxPool2(float*, float*)
                    0.44%  103.71us         1  103.71us  103.71us  103.71us  Linear3_ReLu(float*, float*, float*, float*)
                    0.26%  61.118us         1  61.118us  61.118us  61.118us  [CUDA memcpy DtoH]
      API calls:   95.13%  776.41ms        18  43.134ms  2.3000us  772.40ms  cudaMalloc
                    3.99%  32.525ms        12  2.7104ms  58.200us  15.910ms  cudaMemcpy
                    0.57%  4.6259ms        18  256.99us  2.5000us  1.3957ms  cudaFree
                    0.31%  2.5440ms         1  2.5440ms  2.5440ms  2.5440ms  cuDeviceGetPCIBusId
                    0.01%  47.300us         7  6.7570us  3.0000us  24.600us  cudaLaunchKernel
                    0.00%  17.400us       101     172ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  1.2000us         1  1.2000us  1.2000us  1.2000us  cuDeviceGetName
                    0.00%     900ns         3     300ns     200ns     500ns  cuDeviceGetCount
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceTotalMem
                    0.00%     400ns         1     400ns     400ns     400ns  cuModuleGetLoadingMode
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
```

### 使用锁页内存

```bash
Profiling application: ./infer
Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   41.11%  8.3640ms         1  8.3640ms  8.3640ms  8.3640ms  Conv2_ReLu(float*, float*, float*, float*)
                   23.97%  4.8775ms        11  443.41us  1.2800us  4.8399ms  [CUDA memcpy HtoD]
                   15.53%  3.1592ms         1  3.1592ms  3.1592ms  3.1592ms  Linear1_ReLu(float*, float*, float*, float*)
                    9.44%  1.9200ms         1  1.9200ms  1.9200ms  1.9200ms  Conv1_ReLu(float*, float*, float*, float*)
                    4.56%  928.56us         1  928.56us  928.56us  928.56us  Linear2_ReLu(float*, float*, float*, float*)
                    3.53%  718.60us         1  718.60us  718.60us  718.60us  MaxPool1(float*, float*)
                    1.05%  214.57us         1  214.57us  214.57us  214.57us  MaxPool2(float*, float*)
                    0.51%  104.61us         1  104.61us  104.61us  104.61us  Linear3_ReLu(float*, float*, float*, float*)
                    0.30%  60.515us         1  60.515us  60.515us  60.515us  [CUDA memcpy DtoH]
      API calls:   94.19%  637.19ms         2  318.59ms  882.00us  636.31ms  cudaHostAlloc
                    4.40%  29.785ms        12  2.4821ms  50.600us  15.540ms  cudaMemcpy
                    0.63%  4.2867ms        18  238.15us  2.5000us  1.2993ms  cudaFree
                    0.50%  3.4124ms        18  189.58us  2.1000us  877.20us  cudaMalloc
                    0.25%  1.7246ms         1  1.7246ms  1.7246ms  1.7246ms  cuDeviceGetPCIBusId
                    0.01%  47.300us         7  6.7570us  2.9000us  25.200us  cudaLaunchKernel
                    0.00%  17.800us       101     176ns     100ns     900ns  cuDeviceGetAttribute
                    0.00%     900ns         3     300ns     100ns     600ns  cuDeviceGetCount
                    0.00%     900ns         2     450ns     200ns     700ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuModuleGetLoadingMode
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
```

### 全连接 kernel 访存合并 

```bash
==426896== Profiling application: ./infer
==426896== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   44.47%  8.2537ms         1  8.2537ms  8.2537ms  8.2537ms  Conv2_ReLu(float*, float*, float*, float*)
                   26.24%  4.8705ms        11  442.77us  1.2800us  4.8331ms  [CUDA memcpy HtoD]
                   10.66%  1.9780ms         1  1.9780ms  1.9780ms  1.9780ms  Conv1_ReLu(float*, float*, float*, float*)
                    8.40%  1.5597ms         1  1.5597ms  1.5597ms  1.5597ms  Linear1_ReLu(float*, float*, float*, float*)
                    4.31%  799.88us         1  799.88us  799.88us  799.88us  Linear2_ReLu(float*, float*, float*, float*)
                    3.87%  718.12us         1  718.12us  718.12us  718.12us  MaxPool1(float*, float*)
                    1.16%  214.50us         1  214.50us  214.50us  214.50us  MaxPool2(float*, float*)
                    0.50%  93.216us         1  93.216us  93.216us  93.216us  Linear3_ReLu(float*, float*, float*, float*)
                    0.40%  73.441us         1  73.441us  73.441us  73.441us  [CUDA memcpy DtoH]
      API calls:   94.53%  634.53ms         2  317.26ms  804.20us  633.72ms  cudaHostAlloc
                    4.04%  27.104ms        12  2.2587ms  32.100us  13.775ms  cudaMemcpy
                    0.59%  3.9320ms        18  218.44us  2.3000us  1.1263ms  cudaMalloc
                    0.53%  3.5893ms        18  199.41us  2.3000us  1.1967ms  cudaFree
                    0.31%  2.0558ms         1  2.0558ms  2.0558ms  2.0558ms  cuDeviceGetPCIBusId
                    0.01%  44.300us         7  6.3280us  2.8000us  22.700us  cudaLaunchKernel
                    0.00%  14.200us       101     140ns     100ns     800ns  cuDeviceGetAttribute
                    0.00%  1.0000us         2     500ns     200ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     700ns         3     233ns     100ns     400ns  cuDeviceGetCount
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuModuleGetLoadingMode
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
```

### 计算访存重叠 + kernel fusion

```bash
Profiling application: ./infer
Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   42.50%  12.128ms        16  757.97us  628.96us  920.99us  Conv2_ReLu(float*, float*, float*, float*, float*)
                   20.33%  5.8015ms        16  362.60us  216.58us  595.36us  Linear_ReLu(float*, float*, float*, float*, float*, float*, float*, float*)
                   18.50%  5.2783ms        26  203.01us  1.2800us  340.61us  [CUDA memcpy HtoD]
                   18.27%  5.2133ms        16  325.83us  179.52us  534.27us  Conv1_ReLu(float*, float*, float*, float*, float*)
                    0.40%  113.06us        16  7.0660us  4.3200us  13.728us  [CUDA memcpy DtoH]
      API calls:   95.51%  789.26ms         2  394.63ms  1.3249ms  787.93ms  cudaHostAlloc
                    1.59%  13.136ms         6  2.1893ms     900ns  12.741ms  cudaStreamSynchronize
                    0.96%  7.9675ms        10  796.75us  37.400us  3.7533ms  cudaMemcpy
                    0.55%  4.5692ms         2  2.2846ms  530.70us  4.0385ms  cudaFreeHost
                    0.49%  4.0172ms         1  4.0172ms  4.0172ms  4.0172ms  cuDeviceGetPCIBusId
                    0.48%  3.9615ms        16  247.59us  2.9000us  1.3337ms  cudaMalloc
                    0.32%  2.6745ms        16  167.16us  2.3000us  962.40us  cudaFree
                    0.06%  524.70us        48  10.931us  2.8000us  162.70us  cudaLaunchKernel
                    0.03%  215.10us        32  6.7210us  2.3000us  24.200us  cudaMemcpyAsync
                    0.00%  33.000us         6  5.5000us  1.7000us  23.200us  cudaStreamCreate
                    0.00%  17.800us       101     176ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  16.000us         6  2.6660us  1.2000us  9.3000us  cudaStreamDestroy
                    0.00%  1.4000us         2     700ns     200ns  1.2000us  cuDeviceGet
                    0.00%  1.1000us         3     366ns     200ns     700ns  cuDeviceGetCount
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuModuleGetLoadingMode
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

