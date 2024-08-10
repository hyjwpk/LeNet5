#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

// #define CHECK(call)                                                \
//     {                                                              \
//         const cudaError_t error = call;                            \
//         if (error != cudaSuccess) {                                \
//             fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
//             fprintf(stderr, "code: %d, reason: %s\n", error,       \
//                     cudaGetErrorString(error));                    \
//             exit(1);                                               \
//         }                                                          \
//     }
#define CHECK(call) call

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

float *read_mnist_images(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    file.read((char *)&num_images, sizeof(num_images));
    file.read((char *)&num_rows, sizeof(num_rows));
    file.read((char *)&num_cols, sizeof(num_cols));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                   ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_images = ((num_images & 0xff000000) >> 24) | ((num_images & 0x00ff0000) >> 8) |
                 ((num_images & 0x0000ff00) << 8) | ((num_images & 0x000000ff) << 24);
    num_rows = ((num_rows & 0xff000000) >> 24) | ((num_rows & 0x00ff0000) >> 8) |
               ((num_rows & 0x0000ff00) << 8) | ((num_rows & 0x000000ff) << 24);
    num_cols = ((num_cols & 0xff000000) >> 24) | ((num_cols & 0x00ff0000) >> 8) |
               ((num_cols & 0x0000ff00) << 8) | ((num_cols & 0x000000ff) << 24);

    int image_size = num_rows * num_cols;
    float *images;
    CHECK(cudaMallocHost((void **)&images, num_images * image_size * sizeof(float), cudaHostAllocDefault));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char *)&pixel, sizeof(pixel));
            images[OFFSET(i, j, image_size)] = static_cast<float>(pixel) / 255.0f;
        }
    }

    return images;
}

std::vector<int> read_mnist_labels(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0, num_items = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    file.read((char *)&num_items, sizeof(num_items));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
                   ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_items = ((num_items & 0xff000000) >> 24) | ((num_items & 0x00ff0000) >> 8) |
                ((num_items & 0x0000ff00) << 8) | ((num_items & 0x000000ff) << 24);

    std::vector<int> labels(num_items);
    for (int i = 0; i < num_items; ++i) {
        unsigned char label = 0;
        file.read((char *)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}

std::vector<float> read_param(const std::string &path) {
    std::ifstream file(path);
    std::vector<float> params;
    float param;
    while (file >> param) {
        params.push_back(param);
    }
    return params;
}

__global__ void Conv1_ReLu(float *input_image, float *output_image_1, float *output_image_2, float *kernel_weights, float *kernel_bias) {
    __shared__ float input_image_shared[28][28];
    __shared__ float kernel_weights_shared[5][5];

    for (int i = 0; i < 6; i++) {
        int pos = threadIdx.x * 4 + i * 128;
        float4 image_data = ((float4 *)(input_image + blockIdx.x * 28 * 28 + pos))[0];
        input_image_shared[pos / 28][pos % 28] = image_data.x;
        input_image_shared[(pos + 1) / 28][(pos + 1) % 28] = image_data.y;
        input_image_shared[(pos + 2) / 28][(pos + 2) % 28] = image_data.z;
        input_image_shared[(pos + 3) / 28][(pos + 3) % 28] = image_data.w;
    }
    if (threadIdx.x < 16) {
        int pos = threadIdx.x + 6 * 128;
        float image_data = ((float *)(input_image + blockIdx.x * 28 * 28 + pos))[0];
        input_image_shared[pos / 28][pos % 28] = image_data;
    }

    for (int ii = 0; ii < 6; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            for (int o = 0; o < 6; o++) {
                if (threadIdx.x < 25) {
                    int pos = threadIdx.x;
                    float kernel_data = ((float *)(kernel_weights + o * 5 * 5 + pos))[0];
                    kernel_weights_shared[pos / 5][pos % 5] = kernel_data;
                }

                float value = 0;
                for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < 5; j++) {
                        value += input_image_shared[threadIdx.x / 8 + ii * 4 + i][threadIdx.x % 8 + jj * 8 + j] * kernel_weights_shared[i][j];
                    }
                }
                value += kernel_bias[o];
                value = value > 0 ? value : 0;
                output_image_1[blockIdx.x * 6 * 24 * 24 + o * 24 * 24 + OFFSET(threadIdx.x / 8 + ii * 4, threadIdx.x % 8 + jj * 8, 24)] = value;
            }
        }
    }

    for (int o = 0; o < 6; o++) {
        float value;
        float2 image_data0, image_data1;

        image_data0 = ((float2 *)(output_image_1 + blockIdx.x * 6 * 24 * 24 + o * 24 * 24 + OFFSET(threadIdx.x / 8 * 2, threadIdx.x % 8 * 2, 24)))[0];
        image_data1 = ((float2 *)(output_image_1 + blockIdx.x * 6 * 24 * 24 + o * 24 * 24 + OFFSET(threadIdx.x / 8 * 2 + 1, threadIdx.x % 8 * 2, 24)))[0];
        value = max(max(image_data0.x, image_data0.y), max(image_data1.x, image_data1.y));
        output_image_2[blockIdx.x * 6 * 12 * 12 + o * 12 * 12 + OFFSET(threadIdx.x / 8, threadIdx.x % 8, 12)] = value;

        image_data0 = ((float2 *)(output_image_1 + blockIdx.x * 6 * 24 * 24 + o * 24 * 24 + OFFSET((threadIdx.x / 8 + 4) * 2, threadIdx.x % 8 * 2, 24)))[0];
        image_data1 = ((float2 *)(output_image_1 + blockIdx.x * 6 * 24 * 24 + o * 24 * 24 + OFFSET((threadIdx.x / 8 + 4) * 2 + 1, threadIdx.x % 8 * 2, 24)))[0];
        value = max(max(image_data0.x, image_data0.y), max(image_data1.x, image_data1.y));
        output_image_2[blockIdx.x * 6 * 12 * 12 + o * 12 * 12 + OFFSET(threadIdx.x / 8 + 4, threadIdx.x % 8, 12)] = value;

        image_data0 = ((float2 *)(output_image_1 + blockIdx.x * 6 * 24 * 24 + o * 24 * 24 + OFFSET((threadIdx.x / 8 + 8) * 2, threadIdx.x % 8 * 2, 24)))[0];
        image_data1 = ((float2 *)(output_image_1 + blockIdx.x * 6 * 24 * 24 + o * 24 * 24 + OFFSET((threadIdx.x / 8 + 8) * 2 + 1, threadIdx.x % 8 * 2, 24)))[0];
        value = max(max(image_data0.x, image_data0.y), max(image_data1.x, image_data1.y));
        output_image_2[blockIdx.x * 6 * 12 * 12 + o * 12 * 12 + OFFSET(threadIdx.x / 8 + 8, threadIdx.x % 8, 12)] = value;

        image_data0 = ((float2 *)(output_image_1 + blockIdx.x * 6 * 24 * 24 + o * 24 * 24 + OFFSET(threadIdx.x / 4 * 2, (threadIdx.x % 4 + 8) * 2, 24)))[0];
        image_data1 = ((float2 *)(output_image_1 + blockIdx.x * 6 * 24 * 24 + o * 24 * 24 + OFFSET(threadIdx.x / 4 * 2 + 1, (threadIdx.x % 4 + 8) * 2, 24)))[0];
        value = max(max(image_data0.x, image_data0.y), max(image_data1.x, image_data1.y));
        output_image_2[blockIdx.x * 6 * 12 * 12 + o * 12 * 12 + OFFSET(threadIdx.x / 4, threadIdx.x % 4 + 8, 12)] = value;

        if (threadIdx.x < 16) {
            image_data0 = ((float2 *)(output_image_1 + blockIdx.x * 6 * 24 * 24 + o * 24 * 24 + OFFSET((threadIdx.x / 4 + 8) * 2, (threadIdx.x % 4 + 8) * 2, 24)))[0];
            image_data1 = ((float2 *)(output_image_1 + blockIdx.x * 6 * 24 * 24 + o * 24 * 24 + OFFSET((threadIdx.x / 4 + 8) * 2 + 1, (threadIdx.x % 4 + 8) * 2, 24)))[0];
            value = max(max(image_data0.x, image_data0.y), max(image_data1.x, image_data1.y));
            output_image_2[blockIdx.x * 6 * 12 * 12 + o * 12 * 12 + OFFSET(threadIdx.x / 4 + 8, threadIdx.x % 4 + 8, 12)] = value;
        }
    }
}

__global__ void Conv2_ReLu(float *input_image, float *output_image_1, float *output_image_2, float *kernel_weights, float *kernel_bias) {
    __shared__ float input_image_shared[6][12][12];
    __shared__ float kernel_weights_shared[6][5][5];

    for (int i = 0; i < 6; i++) {
        int pos = threadIdx.x * 4 + i * 128;
        float4 image_data = ((float4 *)(input_image + blockIdx.x * 6 * 12 * 12 + pos))[0];
        input_image_shared[pos / 144][pos % 144 / 12][pos % 144 % 12] = image_data.x;
        input_image_shared[(pos + 1) / 144][(pos + 1) % 144 / 12][(pos + 1) % 144 % 12] = image_data.y;
        input_image_shared[(pos + 2) / 144][(pos + 2) % 144 / 12][(pos + 2) % 144 % 12] = image_data.z;
        input_image_shared[(pos + 3) / 144][(pos + 3) % 144 / 12][(pos + 3) % 144 % 12] = image_data.w;
    }
    {
        int pos = threadIdx.x * 2 + 6 * 128;
        float2 image_data = ((float2 *)(input_image + blockIdx.x * 6 * 12 * 12 + pos))[0];
        input_image_shared[pos / 144][pos % 144 / 12][pos % 144 % 12] = image_data.x;
        input_image_shared[(pos + 1) / 144][(pos + 1) % 144 / 12][(pos + 1) % 144 % 12] = image_data.y;
    }
    {
        int pos = threadIdx.x + 6 * 128 + 64;
        float image_data = ((float *)(input_image + blockIdx.x * 6 * 12 * 12 + pos))[0];
        input_image_shared[pos / 144][pos % 144 / 12][pos % 144 % 12] = image_data;
    }

    for (int o = 0; o < 16; o++) {
        {
            int pos = threadIdx.x * 2;
            float2 kernel_data = ((float2 *)(kernel_weights + o * 6 * 5 * 5 + pos))[0];
            kernel_weights_shared[pos / 25][pos % 25 / 5][pos % 25 % 5] = kernel_data.x;
            kernel_weights_shared[(pos + 1) / 25][(pos + 1) % 25 / 5][(pos + 1) % 25 % 5] = kernel_data.y;
        }
        {
            int pos = threadIdx.x * 2 + 64;
            float2 kernel_data = ((float2 *)(kernel_weights + o * 6 * 5 * 5 + pos))[0];
            kernel_weights_shared[pos / 25][pos % 25 / 5][pos % 25 % 5] = kernel_data.x;
            kernel_weights_shared[(pos + 1) / 25][(pos + 1) % 25 / 5][(pos + 1) % 25 % 5] = kernel_data.y;
        }
        if (threadIdx.x < 22) {
            int pos = threadIdx.x + 128;
            float kernel_data = ((float *)(kernel_weights + o * 6 * 5 * 5 + pos))[0];
            kernel_weights_shared[pos / 25][pos % 25 / 5][pos % 25 % 5] = kernel_data;
        }

        float bias = kernel_bias[o];
        float value = 0;
        for (int z = 0; z < 6; z++) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    value += input_image_shared[z][threadIdx.x / 8 + i][threadIdx.x % 8 + j] * kernel_weights_shared[z][i][j];
                }
            }
        }
        value += bias;
        value = value > 0 ? value : 0;
        output_image_1[blockIdx.x * 16 * 8 * 8 + o * 8 * 8 + OFFSET(threadIdx.x / 8, threadIdx.x % 8, 8)] = value;

        value = 0;
        for (int z = 0; z < 6; z++) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    value += input_image_shared[z][threadIdx.x / 8 + 4 + i][threadIdx.x % 8 + j] * kernel_weights_shared[z][i][j];
                }
            }
        }
        value += bias;
        value = value > 0 ? value : 0;
        output_image_1[blockIdx.x * 16 * 8 * 8 + o * 8 * 8 + OFFSET(threadIdx.x / 8 + 4, threadIdx.x % 8, 8)] = value;
    }

    int channel = threadIdx.x / 4;
    int y = threadIdx.x % 4 / 2;
    int x = threadIdx.x % 4 % 2;
    float value0 = 0, value1 = 0, value2 = 0, value3 = 0;
    float4 image_data0 = ((float4 *)(output_image_1 + blockIdx.x * 16 * 8 * 8 + channel * 8 * 8 + OFFSET(y * 4, x * 4, 8)))[0];
    float4 image_data1 = ((float4 *)(output_image_1 + blockIdx.x * 16 * 8 * 8 + channel * 8 * 8 + OFFSET(y * 4 + 1, x * 4, 8)))[0];
    float4 image_data2 = ((float4 *)(output_image_1 + blockIdx.x * 16 * 8 * 8 + channel * 8 * 8 + OFFSET(y * 4 + 2, x * 4, 8)))[0];
    float4 image_data3 = ((float4 *)(output_image_1 + blockIdx.x * 16 * 8 * 8 + channel * 8 * 8 + OFFSET(y * 4 + 3, x * 4, 8)))[0];
    value0 = max(max(image_data0.x, image_data0.y), max(image_data1.x, image_data1.y));
    value1 = max(max(image_data0.z, image_data0.w), max(image_data1.z, image_data1.w));
    value2 = max(max(image_data2.x, image_data2.y), max(image_data3.x, image_data3.y));
    value3 = max(max(image_data2.z, image_data2.w), max(image_data3.z, image_data3.w));
    output_image_2[blockIdx.x * 16 * 4 * 4 + channel * 4 * 4 + OFFSET(y * 2, x * 2, 4)] = value0;
    output_image_2[blockIdx.x * 16 * 4 * 4 + channel * 4 * 4 + OFFSET(y * 2, x * 2 + 1, 4)] = value1;
    output_image_2[blockIdx.x * 16 * 4 * 4 + channel * 4 * 4 + OFFSET(y * 2 + 1, x * 2, 4)] = value2;
    output_image_2[blockIdx.x * 16 * 4 * 4 + channel * 4 * 4 + OFFSET(y * 2 + 1, x * 2 + 1, 4)] = value3;

    channel = threadIdx.x / 4 + 8;
    image_data0 = ((float4 *)(output_image_1 + blockIdx.x * 16 * 8 * 8 + channel * 8 * 8 + OFFSET(y * 4, x * 4, 8)))[0];
    image_data1 = ((float4 *)(output_image_1 + blockIdx.x * 16 * 8 * 8 + channel * 8 * 8 + OFFSET(y * 4 + 1, x * 4, 8)))[0];
    image_data2 = ((float4 *)(output_image_1 + blockIdx.x * 16 * 8 * 8 + channel * 8 * 8 + OFFSET(y * 4 + 2, x * 4, 8)))[0];
    image_data3 = ((float4 *)(output_image_1 + blockIdx.x * 16 * 8 * 8 + channel * 8 * 8 + OFFSET(y * 4 + 3, x * 4, 8)))[0];
    value0 = max(max(image_data0.x, image_data0.y), max(image_data1.x, image_data1.y));
    value1 = max(max(image_data0.z, image_data0.w), max(image_data1.z, image_data1.w));
    value2 = max(max(image_data2.x, image_data2.y), max(image_data3.x, image_data3.y));
    value3 = max(max(image_data2.z, image_data2.w), max(image_data3.z, image_data3.w));
    output_image_2[blockIdx.x * 16 * 4 * 4 + channel * 4 * 4 + OFFSET(y * 2, x * 2, 4)] = value0;
    output_image_2[blockIdx.x * 16 * 4 * 4 + channel * 4 * 4 + OFFSET(y * 2, x * 2 + 1, 4)] = value1;
    output_image_2[blockIdx.x * 16 * 4 * 4 + channel * 4 * 4 + OFFSET(y * 2 + 1, x * 2, 4)] = value2;
    output_image_2[blockIdx.x * 16 * 4 * 4 + channel * 4 * 4 + OFFSET(y * 2 + 1, x * 2 + 1, 4)] = value3;
}

template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (blockSize >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (blockSize >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (blockSize >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (blockSize >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

__global__ void Linear_ReLu(float *input_image, float *output_image, float *Linear1_kernel_weights, float *Linear1_kernel_bias, float *Linear2_kernel_weights, float *Linear2_kernel_bias, float *Linear3_kernel_weights, float *Linear3_kernel_bias) {
    __shared__ float output_image_shared1[120];
    __shared__ float output_image_shared2[84];

    float4 image_data0 = ((float4 *)(input_image + blockIdx.x * 256 + threadIdx.x * 4))[0];
    float4 image_data1 = ((float4 *)(input_image + blockIdx.x * 256 + threadIdx.x * 4 + 128))[0];
    for (int i = 0; i < 120; i++) {
        float value = 0;
        float4 kernel_data0 = ((float4 *)(Linear1_kernel_weights + OFFSET(i, threadIdx.x * 4, 256)))[0];
        float4 kernel_data1 = ((float4 *)(Linear1_kernel_weights + OFFSET(i, threadIdx.x * 4 + 128, 256)))[0];
        value += image_data0.x * kernel_data0.x + image_data0.y * kernel_data0.y + image_data0.z * kernel_data0.z + image_data0.w * kernel_data0.w;
        value += image_data1.x * kernel_data1.x + image_data1.y * kernel_data1.y + image_data1.z * kernel_data1.z + image_data1.w * kernel_data1.w;
        value = warpReduceSum<32>(value);
        if (threadIdx.x == 0) {
            value += Linear1_kernel_bias[i];
            value = value > 0 ? value : 0;
            output_image_shared1[i] = value;
        }
    }

    for (int i = 0; i < 84; i++) {
        float value = 0;
        float2 kernel_data0 = ((float2 *)(Linear2_kernel_weights + OFFSET(i, threadIdx.x * 2, 120)))[0];
        float kernel_data1 = ((float *)(Linear2_kernel_weights + OFFSET(i, threadIdx.x + 64, 120)))[0];
        float kernel_data2 = 0;
        if (threadIdx.x < 24) {
            kernel_data2 = ((float *)(Linear2_kernel_weights + OFFSET(i, threadIdx.x + 96, 120)))[0];
        }
        value += output_image_shared1[threadIdx.x * 2] * kernel_data0.x + output_image_shared1[threadIdx.x * 2 + 1] * kernel_data0.y + output_image_shared1[threadIdx.x + 64] * kernel_data1;
        if (threadIdx.x < 24) {
            value += output_image_shared1[threadIdx.x + 96] * kernel_data2;
        }
        value = warpReduceSum<32>(value);
        if (threadIdx.x == 0) {
            value += Linear2_kernel_bias[i];
            value = value > 0 ? value : 0;
            output_image_shared2[i] = value;
        }
    }

    for (int i = 0; i < 10; i++) {
        float value = 0;
        float2 kernel_data0 = ((float2 *)(Linear3_kernel_weights + OFFSET(i, threadIdx.x * 2, 84)))[0];
        float kernel_data1 = 0;
        if (threadIdx.x < 20) {
            kernel_data1 = ((float *)(Linear3_kernel_weights + OFFSET(i, threadIdx.x + 64, 84)))[0];
        }
        value += output_image_shared2[threadIdx.x * 2] * kernel_data0.x + output_image_shared2[threadIdx.x * 2 + 1] * kernel_data0.y;
        if (threadIdx.x < 20) {
            value += output_image_shared2[threadIdx.x + 64] * kernel_data1;
        }
        value = warpReduceSum<32>(value);
        if (threadIdx.x == 0) {
            value += Linear3_kernel_bias[i];
            output_image[blockIdx.x * 10 + i] = value;
        }
    }
}

int main(int argc, char *argv[]) {
    auto images = read_mnist_images("./data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels("./data/FashionMNIST/raw/t10k-labels-idx1-ubyte");

    auto Conv1_weight = read_param("./weight/conv1.weight.txt");
    auto Conv1_bias = read_param("./weight/conv1.bias.txt");
    auto Conv2_weight = read_param("./weight/conv2.weight.txt");
    auto Conv2_bias = read_param("./weight/conv2.bias.txt");
    auto Linear1_weight = read_param("./weight/fc1.weight.txt");
    auto Linear1_bias = read_param("./weight/fc1.bias.txt");
    auto Linear2_weight = read_param("./weight/fc2.weight.txt");
    auto Linear2_bias = read_param("./weight/fc2.bias.txt");
    auto Linear3_weight = read_param("./weight/fc3.weight.txt");
    auto Linear3_bias = read_param("./weight/fc3.bias.txt");

    // Conv1 [1, 28, 28] -> [6, 24, 24]
    const int batchsize = 10000;
    const int stream_size = 625;
    const int Conv1_input_height = 28;
    const int Conv1_kernel_height = 5;
    const int Conv1_output_height = Conv1_input_height - Conv1_kernel_height + 1;
    const int Conv1_input_channel = 1;
    const int Conv1_output_channel = 6;
    float *device_input_image;
    float *device_Conv1_output_image;
    float *device_Conv1_kernel_weight;
    float *device_Conv1_kernel_bias;
    CHECK(cudaMalloc((void **)&device_input_image, batchsize * Conv1_input_channel * Conv1_input_height * Conv1_input_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Conv1_output_image, batchsize * Conv1_output_channel * Conv1_output_height * Conv1_output_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Conv1_kernel_weight, Conv1_output_channel * Conv1_input_channel * Conv1_kernel_height * Conv1_kernel_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Conv1_kernel_bias, Conv1_output_channel * sizeof(float)));
    CHECK(cudaMemcpy(device_Conv1_kernel_weight, &Conv1_weight[0], Conv1_output_channel * Conv1_input_channel * Conv1_kernel_height * Conv1_kernel_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Conv1_kernel_bias, &Conv1_bias[0], Conv1_output_channel * sizeof(float), cudaMemcpyHostToDevice));
    dim3 Conv1_blocksperGrid(stream_size);
    dim3 Conv1_threadsperBlock(32);

    // MaxPool1 [6, 24, 24] -> [6, 12, 12]
    const int MaxPool1_input_height = Conv1_output_height;
    const int MaxPool1_output_height = MaxPool1_input_height / 2;
    const int MaxPool1_channel = Conv1_output_channel;
    float *device_MaxPool1_output_image;
    CHECK(cudaMalloc((void **)&device_MaxPool1_output_image, batchsize * MaxPool1_channel * MaxPool1_output_height * MaxPool1_output_height * sizeof(float)));

    // Conv2 [6, 12, 12] -> [16, 8, 8]
    const int Conv2_input_height = MaxPool1_output_height;
    const int Conv2_kernel_height = 5;
    const int Conv2_output_height = Conv2_input_height - Conv2_kernel_height + 1;
    const int Conv2_input_channel = 6;
    const int Conv2_output_channel = 16;
    float *device_Conv2_output_image;
    float *device_Conv2_kernel_weight;
    float *device_Conv2_kernel_bias;
    CHECK(cudaMalloc((void **)&device_Conv2_output_image, batchsize * Conv2_output_channel * Conv2_output_height * Conv2_output_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Conv2_kernel_weight, Conv2_output_channel * Conv2_input_channel * Conv2_kernel_height * Conv2_kernel_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Conv2_kernel_bias, Conv2_output_channel * sizeof(float)));
    CHECK(cudaMemcpy(device_Conv2_kernel_weight, &Conv2_weight[0], Conv2_output_channel * Conv2_input_channel * Conv2_kernel_height * Conv2_kernel_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Conv2_kernel_bias, &Conv2_bias[0], Conv2_output_channel * sizeof(float), cudaMemcpyHostToDevice));
    dim3 Conv2_blocksperGrid(stream_size);
    dim3 Conv2_threadsperBlock(32);

    // MaxPool2 [16, 8, 8] -> [16, 4, 4]
    const int MaxPool2_input_height = Conv2_output_height;
    const int MaxPool2_output_height = MaxPool2_input_height / 2;
    const int MaxPool2_channel = Conv2_output_channel;
    float *device_MaxPool2_output_image;
    CHECK(cudaMalloc((void **)&device_MaxPool2_output_image, batchsize * MaxPool2_channel * MaxPool2_output_height * MaxPool2_output_height * sizeof(float)));

    // Linear1 [256] -> [120]
    const int Linear1_input_height = MaxPool2_channel * MaxPool2_output_height * MaxPool2_output_height;
    const int Linear1_output_height = 120;
    float *device_Linear1_kernel_weight;
    float *device_Linear1_kernel_bias;
    CHECK(cudaMalloc((void **)&device_Linear1_kernel_weight, Linear1_output_height * Linear1_input_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear1_kernel_bias, Linear1_output_height * sizeof(float)));
    CHECK(cudaMemcpy(device_Linear1_kernel_weight, &Linear1_weight[0], Linear1_output_height * Linear1_input_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Linear1_kernel_bias, &Linear1_bias[0], Linear1_output_height * sizeof(float), cudaMemcpyHostToDevice));
    dim3 Linear1_blocksperGrid(stream_size);
    dim3 Linear1_threadsperBlock(32);

    // Linear2 [120] -> [84]
    const int Linear2_input_height = Linear1_output_height;
    const int Linear2_output_height = 84;
    float *device_Linear2_kernel_weight;
    float *device_Linear2_kernel_bias;
    CHECK(cudaMalloc((void **)&device_Linear2_kernel_weight, Linear2_output_height * Linear2_input_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear2_kernel_bias, Linear2_output_height * sizeof(float)));
    CHECK(cudaMemcpy(device_Linear2_kernel_weight, &Linear2_weight[0], Linear2_output_height * Linear2_input_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Linear2_kernel_bias, &Linear2_bias[0], Linear2_output_height * sizeof(float), cudaMemcpyHostToDevice));

    // Linear3 [84] -> [10]
    const int Linear3_input_height = Linear2_output_height;
    const int Linear3_output_height = 10;
    float *device_Linear3_output_image;
    float *device_Linear3_kernel_weight;
    float *device_Linear3_kernel_bias;
    float *host_Linear3_output_image;
    CHECK(cudaMalloc((void **)&device_Linear3_output_image, batchsize * Linear3_output_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear3_kernel_weight, Linear3_output_height * Linear3_input_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear3_kernel_bias, Linear3_output_height * sizeof(float)));
    CHECK(cudaMemcpy(device_Linear3_kernel_weight, &Linear3_weight[0], Linear3_output_height * Linear3_input_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Linear3_kernel_bias, &Linear3_bias[0], Linear3_output_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMallocHost((void **)&host_Linear3_output_image, batchsize * Linear3_output_height * sizeof(float), cudaHostAllocDefault));

    const int nStreams = 6;
    // Stream的初始化
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int correct_nums = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < batchsize / stream_size; i++) {
        int stream_id = i % nStreams;
        CHECK(cudaMemcpyAsync(device_input_image + i * stream_size * Conv1_input_height * Conv1_input_height, images + i * stream_size * Conv1_input_height * Conv1_input_height, stream_size * Conv1_input_height * Conv1_input_height * sizeof(float), cudaMemcpyHostToDevice, streams[stream_id]));
        // Conv1 + MaxPool1
        Conv1_ReLu<<<Conv1_blocksperGrid, Conv1_threadsperBlock, 0, streams[stream_id]>>>(device_input_image + i * stream_size * Conv1_input_height * Conv1_input_height, device_Conv1_output_image + i * stream_size * Conv1_output_channel * Conv1_output_height * Conv1_output_height, device_MaxPool1_output_image + i * stream_size * MaxPool1_channel * MaxPool1_output_height * MaxPool1_output_height, device_Conv1_kernel_weight, device_Conv1_kernel_bias);
        // Conv2 + MaxPool2
        Conv2_ReLu<<<Conv2_blocksperGrid, Conv2_threadsperBlock, 0, streams[stream_id]>>>(device_MaxPool1_output_image + i * stream_size * MaxPool1_channel * MaxPool1_output_height * MaxPool1_output_height, device_Conv2_output_image + i * stream_size * Conv2_output_channel * Conv2_output_height * Conv2_output_height, device_MaxPool2_output_image + i * stream_size * MaxPool2_channel * MaxPool2_output_height * MaxPool2_output_height, device_Conv2_kernel_weight, device_Conv2_kernel_bias);
        // Linear1 + Linear2 + Linear3
        Linear_ReLu<<<Linear1_blocksperGrid, Linear1_threadsperBlock, 0, streams[stream_id]>>>(device_MaxPool2_output_image + i * stream_size * MaxPool2_channel * MaxPool2_output_height * MaxPool2_output_height, device_Linear3_output_image + i * stream_size * Linear3_output_height, device_Linear1_kernel_weight, device_Linear1_kernel_bias, device_Linear2_kernel_weight, device_Linear2_kernel_bias, device_Linear3_kernel_weight, device_Linear3_kernel_bias);
        CHECK(cudaMemcpyAsync(host_Linear3_output_image + i * stream_size * Linear3_output_height, device_Linear3_output_image + i * stream_size * Linear3_output_height, stream_size * Linear3_output_height * sizeof(float), cudaMemcpyDeviceToHost, streams[stream_id]));
    }
    for (int i = 0; i < nStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    for (int t = 0; t < batchsize; t++) {
        int index = 0;
        for (int k = 0; k < 10; k++) {
            if (host_Linear3_output_image[t * Linear3_output_height + k] > host_Linear3_output_image[t * Linear3_output_height + index]) {
                index = k;
            }
        }
        if (index == labels[t]) {
            correct_nums++;
        }
    }

    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << float(correct_nums) / float(batchsize) << std::endl;

    CHECK(cudaFree(device_input_image));
    // Conv1
    CHECK(cudaFree(device_Conv1_output_image));
    CHECK(cudaFree(device_Conv1_kernel_weight));
    CHECK(cudaFree(device_Conv1_kernel_bias));
    // MaxPool1
    CHECK(cudaFree(device_MaxPool1_output_image));
    // Conv2
    CHECK(cudaFree(device_Conv2_output_image));
    CHECK(cudaFree(device_Conv2_kernel_weight));
    CHECK(cudaFree(device_Conv2_kernel_bias));
    // MaxPool2
    CHECK(cudaFree(device_MaxPool2_output_image));
    // Linear1
    CHECK(cudaFree(device_Linear1_kernel_weight));
    CHECK(cudaFree(device_Linear1_kernel_bias));
    // Linear2
    CHECK(cudaFree(device_Linear2_kernel_weight));
    CHECK(cudaFree(device_Linear2_kernel_bias));
    // Linear3
    CHECK(cudaFree(device_Linear3_kernel_weight));
    CHECK(cudaFree(device_Linear3_kernel_bias));
    CHECK(cudaFree(device_Linear3_output_image));
    // Host
    CHECK(cudaFreeHost(images));
    CHECK(cudaFreeHost(host_Linear3_output_image));
    // Stream
    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    return 0;
}