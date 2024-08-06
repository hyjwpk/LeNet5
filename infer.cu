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
    float *images = (float *)malloc(num_images * image_size * sizeof(float));

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

__global__ void Conv1_ReLu(float *input_image, float *output_image, float *kernel_weights, float *kernel_bias) {
    __shared__ float input_image_shared[28][28];
    __shared__ float kernel_weights_shared[6][5][5];
    __shared__ float kernel_bias_shared[6];

    for (int i = 0; i < 3; i++) {
        int pos = (threadIdx.y * 8 + threadIdx.x) * 4 + i * 256;
        float4 image_data = ((float4 *)(input_image + blockIdx.x * 28 * 28 + pos))[0];
        input_image_shared[pos / 28][pos % 28] = image_data.x;
        input_image_shared[(pos + 1) / 28][(pos + 1) % 28] = image_data.y;
        input_image_shared[(pos + 2) / 28][(pos + 2) % 28] = image_data.z;
        input_image_shared[(pos + 3) / 28][(pos + 3) % 28] = image_data.w;
    }
    if (threadIdx.y == 0 && threadIdx.x < 4) {
        int pos = threadIdx.x * 4 + 3 * 256;
        float4 image_data = ((float4 *)(input_image + blockIdx.x * 28 * 28 + pos))[0];
        input_image_shared[pos / 28][pos % 28] = image_data.x;
        input_image_shared[(pos + 1) / 28][(pos + 1) % 28] = image_data.y;
        input_image_shared[(pos + 2) / 28][(pos + 2) % 28] = image_data.z;
        input_image_shared[(pos + 3) / 28][(pos + 3) % 28] = image_data.w;
    }
    if (threadIdx.y * 8 + threadIdx.x < 37) {
        int pos = (threadIdx.y * 8 + threadIdx.x) * 4;
        float4 kernel_data = ((float4 *)(kernel_weights + pos))[0];
        kernel_weights_shared[pos / 25][pos % 25 / 5][pos % 25 % 5] = kernel_data.x;
        kernel_weights_shared[(pos + 1) / 25][(pos + 1) % 25 / 5][(pos + 1) % 25 % 5] = kernel_data.y;
        kernel_weights_shared[(pos + 2) / 25][(pos + 2) % 25 / 5][(pos + 2) % 25 % 5] = kernel_data.z;
        kernel_weights_shared[(pos + 3) / 25][(pos + 3) % 25 / 5][(pos + 3) % 25 % 5] = kernel_data.w;
    }
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        int pos = 37 * 4;
        float2 kernel_data = ((float2 *)(kernel_weights + pos))[0];
        kernel_weights_shared[pos / 25][pos % 25 / 5][pos % 25 % 5] = kernel_data.x;
        kernel_weights_shared[(pos + 1) / 25][(pos + 1) % 25 / 5][(pos + 1) % 25 % 5] = kernel_data.y;
    }
    if (threadIdx.y == 0 && threadIdx.x < 6) {
        kernel_bias_shared[threadIdx.x] = kernel_bias[threadIdx.x];
    }
    __syncthreads();

    for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            for (int o = 0; o < 6; o++) {
                float value = 0;
                for (int i = 0; i < 5; i++) {
                    for (int j = 0; j < 5; j++) {
                        value += input_image_shared[threadIdx.y * 3 + ii + i][threadIdx.x * 3 + jj + j] * kernel_weights_shared[o][i][j];
                    }
                }
                value += kernel_bias_shared[o];
                value = value > 0 ? value : 0;
                output_image[blockIdx.x * 6 * 24 * 24 + o * 24 * 24 + OFFSET(threadIdx.y * 3 + ii, threadIdx.x * 3 + jj, 24)] = value;
            }
        }
    }
}

__global__ void Conv2_ReLu(float *input_image, float *output_image, float *kernel_weights, float *kernel_bias) {
    __shared__ float input_image_shared[6][12][12];
    __shared__ float kernel_weights_shared[16][6][5][5];
    __shared__ float kernel_bias_shared[16];

    for (int i = 0; i < 3; i++) {
        int pos = (threadIdx.y * 8 + threadIdx.x) * 4 + i * 256;
        float4 image_data = ((float4 *)(input_image + blockIdx.x * 6 * 12 * 12 + pos))[0];
        input_image_shared[pos / 144][pos % 144 / 12][pos % 144 % 12] = image_data.x;
        input_image_shared[(pos + 1) / 144][(pos + 1) % 144 / 12][(pos + 1) % 144 % 12] = image_data.y;
        input_image_shared[(pos + 2) / 144][(pos + 2) % 144 / 12][(pos + 2) % 144 % 12] = image_data.z;
        input_image_shared[(pos + 3) / 144][(pos + 3) % 144 / 12][(pos + 3) % 144 % 12] = image_data.w;
    }
    if (threadIdx.y * 8 + threadIdx.x < 24) {
        int pos = (threadIdx.y * 8 + threadIdx.x) * 4 + 3 * 256;
        float4 image_data = ((float4 *)(input_image + blockIdx.x * 6 * 12 * 12 + pos))[0];
        input_image_shared[pos / 144][pos % 144 / 12][pos % 144 % 12] = image_data.x;
        input_image_shared[(pos + 1) / 144][(pos + 1) % 144 / 12][(pos + 1) % 144 % 12] = image_data.y;
        input_image_shared[(pos + 2) / 144][(pos + 2) % 144 / 12][(pos + 2) % 144 % 12] = image_data.z;
        input_image_shared[(pos + 3) / 144][(pos + 3) % 144 / 12][(pos + 3) % 144 % 12] = image_data.w;
    }
    for (int i = 0; i < 9; i++) {
        int pos = (threadIdx.y * 8 + threadIdx.x) * 4 + i * 256;
        float4 kernel_data = ((float4 *)(kernel_weights + pos))[0];
        kernel_weights_shared[pos / 150][pos % 150 / 25][pos % 150 % 25 / 5][pos % 150 % 25 % 5] = kernel_data.x;
        kernel_weights_shared[(pos + 1) / 150][(pos + 1) % 150 / 25][(pos + 1) % 150 % 25 / 5][(pos + 1) % 150 % 25 % 5] = kernel_data.y;
        kernel_weights_shared[(pos + 2) / 150][(pos + 2) % 150 / 25][(pos + 2) % 150 % 25 / 5][(pos + 2) % 150 % 25 % 5] = kernel_data.z;
        kernel_weights_shared[(pos + 3) / 150][(pos + 3) % 150 / 25][(pos + 3) % 150 % 25 / 5][(pos + 3) % 150 % 25 % 5] = kernel_data.w;
    }
    if (threadIdx.y * 8 + threadIdx.x < 24) {
        int pos = (threadIdx.y * 8 + threadIdx.x) * 4 + 9 * 256;
        float4 kernel_data = ((float4 *)(kernel_weights + pos))[0];
        kernel_weights_shared[pos / 150][pos % 150 / 25][pos % 150 % 25 / 5][pos % 150 % 25 % 5] = kernel_data.x;
        kernel_weights_shared[(pos + 1) / 150][(pos + 1) % 150 / 25][(pos + 1) % 150 % 25 / 5][(pos + 1) % 150 % 25 % 5] = kernel_data.y;
        kernel_weights_shared[(pos + 2) / 150][(pos + 2) % 150 / 25][(pos + 2) % 150 % 25 / 5][(pos + 2) % 150 % 25 % 5] = kernel_data.z;
        kernel_weights_shared[(pos + 3) / 150][(pos + 3) % 150 / 25][(pos + 3) % 150 % 25 / 5][(pos + 3) % 150 % 25 % 5] = kernel_data.w;
    }
    if (threadIdx.y * blockDim.x + threadIdx.x < 16) {
        kernel_bias_shared[threadIdx.y * blockDim.x + threadIdx.x] = kernel_bias[threadIdx.y * blockDim.x + threadIdx.x];
    }
    __syncthreads();

    for (int o = 0; o < 16; o++) {
        float value = 0;
        for (int z = 0; z < 6; z++) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    value += input_image_shared[z][threadIdx.y + i][threadIdx.x + j] * kernel_weights_shared[o][z][i][j];
                }
            }
        }
        value += kernel_bias_shared[o];
        value = value > 0 ? value : 0;
        output_image[blockIdx.x * 16 * 8 * 8 + o * 8 * 8 + OFFSET(threadIdx.y, threadIdx.x, 8)] = value;
    }
}

__global__ void MaxPool(float *input_image, float *output_image, int input_height, int output_height, int channel) {
    float value = 0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int input_image_index = blockIdx.y * channel * input_height * input_height + blockIdx.x * input_height * input_height + OFFSET(threadIdx.y * 2 + i, threadIdx.x * 2 + j, input_height);
            if (input_image[input_image_index] > value) {
                value = input_image[input_image_index];
            }
        }
    }
    output_image[blockIdx.y * channel * output_height * output_height + blockIdx.x * output_height * output_height + threadIdx.y * output_height + threadIdx.x] = value;
}

__global__ void Linear_ReLu(float *input_image, float *output_image, float *kernel_weights, float *kernel_bias, int input_height, int output_height, bool ReLu) {
    float value = 0;
    for (int i = 0; i < input_height; i++) {
        value += input_image[blockIdx.x * input_height + i] * kernel_weights[OFFSET(threadIdx.x, i, input_height)];
    }
    value += kernel_bias[threadIdx.x];
    value = ((!ReLu) || (value > 0)) ? value : 0;
    output_image[blockIdx.x * output_height + threadIdx.x] = value;
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
    dim3 Conv1_blocksperGrid(batchsize);
    dim3 Conv1_threadsperBlock(8, 8);

    // MaxPool1 [6, 24, 24] -> [6, 12, 12]
    const int MaxPool1_input_height = Conv1_output_height;
    const int MaxPool1_output_height = MaxPool1_input_height / 2;
    const int MaxPool1_channel = Conv1_output_channel;
    float *device_MaxPool1_output_image;
    CHECK(cudaMalloc((void **)&device_MaxPool1_output_image, batchsize * MaxPool1_channel * MaxPool1_output_height * MaxPool1_output_height * sizeof(float)));
    dim3 MaxPool1_blocksperGrid(MaxPool1_channel, batchsize);
    dim3 MaxPool1_threadsperBlock(MaxPool1_output_height, MaxPool1_output_height);

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
    dim3 Conv2_blocksperGrid(batchsize);
    dim3 Conv2_threadsperBlock(8, 8);

    // MaxPool2 [16, 8, 8] -> [16, 4, 4]
    const int MaxPool2_input_height = Conv2_output_height;
    const int MaxPool2_output_height = MaxPool2_input_height / 2;
    const int MaxPool2_channel = Conv2_output_channel;
    float *device_MaxPool2_output_image;
    CHECK(cudaMalloc((void **)&device_MaxPool2_output_image, batchsize * MaxPool2_channel * MaxPool2_output_height * MaxPool2_output_height * sizeof(float)));
    dim3 MaxPool2_blocksperGrid(MaxPool2_channel, batchsize);
    dim3 MaxPool2_threadsperBlock(MaxPool2_output_height, MaxPool2_output_height);

    // Linear1 [256] -> [120]
    const int Linear1_input_height = MaxPool2_channel * MaxPool2_output_height * MaxPool2_output_height;
    const int Linear1_output_height = 120;
    float *device_Linear1_output_image;
    float *device_Linear1_kernel_weight;
    float *device_Linear1_kernel_bias;
    CHECK(cudaMalloc((void **)&device_Linear1_output_image, batchsize * Linear1_output_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear1_kernel_weight, Linear1_output_height * Linear1_input_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear1_kernel_bias, Linear1_output_height * sizeof(float)));
    CHECK(cudaMemcpy(device_Linear1_kernel_weight, &Linear1_weight[0], Linear1_output_height * Linear1_input_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Linear1_kernel_bias, &Linear1_bias[0], Linear1_output_height * sizeof(float), cudaMemcpyHostToDevice));
    dim3 Linear1_blocksperGrid(batchsize);
    dim3 Linear1_threadsperBlock(Linear1_output_height);

    // Linear2 [120] -> [84]
    const int Linear2_input_height = Linear1_output_height;
    const int Linear2_output_height = 84;
    float *device_Linear2_output_image;
    float *device_Linear2_kernel_weight;
    float *device_Linear2_kernel_bias;
    CHECK(cudaMalloc((void **)&device_Linear2_output_image, batchsize * Linear2_output_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear2_kernel_weight, Linear2_output_height * Linear2_input_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear2_kernel_bias, Linear2_output_height * sizeof(float)));
    CHECK(cudaMemcpy(device_Linear2_kernel_weight, &Linear2_weight[0], Linear2_output_height * Linear2_input_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Linear2_kernel_bias, &Linear2_bias[0], Linear2_output_height * sizeof(float), cudaMemcpyHostToDevice));
    dim3 Linear2_blocksperGrid(batchsize);
    dim3 Linear2_threadsperBlock(Linear2_output_height);

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
    host_Linear3_output_image = (float *)malloc(batchsize * Linear3_output_height * sizeof(float));
    dim3 Linear3_blocksperGrid(batchsize);
    dim3 Linear3_threadsperBlock(Linear3_output_height);

    int correct_nums = 0;
    auto start = std::chrono::high_resolution_clock::now();

    CHECK(cudaMemcpy(device_input_image, images, batchsize * Conv1_input_height * Conv1_input_height * sizeof(float), cudaMemcpyHostToDevice));
    // Conv1
    Conv1_ReLu<<<Conv1_blocksperGrid, Conv1_threadsperBlock>>>(device_input_image, device_Conv1_output_image, device_Conv1_kernel_weight, device_Conv1_kernel_bias);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    // MaxPool1
    MaxPool<<<MaxPool1_blocksperGrid, MaxPool1_threadsperBlock>>>(device_Conv1_output_image, device_MaxPool1_output_image, MaxPool1_input_height, MaxPool1_output_height, MaxPool1_channel);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    // Conv2
    Conv2_ReLu<<<Conv2_blocksperGrid, Conv2_threadsperBlock>>>(device_MaxPool1_output_image, device_Conv2_output_image, device_Conv2_kernel_weight, device_Conv2_kernel_bias);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    // MaxPool2
    MaxPool<<<MaxPool2_blocksperGrid, MaxPool2_threadsperBlock>>>(device_Conv2_output_image, device_MaxPool2_output_image, MaxPool2_input_height, MaxPool2_output_height, MaxPool2_channel);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    // Linear1
    Linear_ReLu<<<Linear1_blocksperGrid, Linear1_threadsperBlock>>>(device_MaxPool2_output_image, device_Linear1_output_image, device_Linear1_kernel_weight, device_Linear1_kernel_bias, Linear1_input_height, Linear1_output_height, true);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    // Linear2
    Linear_ReLu<<<Linear2_blocksperGrid, Linear2_threadsperBlock>>>(device_Linear1_output_image, device_Linear2_output_image, device_Linear2_kernel_weight, device_Linear2_kernel_bias, Linear2_input_height, Linear2_output_height, true);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    // Linear3
    Linear_ReLu<<<Linear3_blocksperGrid, Linear3_threadsperBlock>>>(device_Linear2_output_image, device_Linear3_output_image, device_Linear3_kernel_weight, device_Linear3_kernel_bias, Linear3_input_height, Linear3_output_height, false);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(host_Linear3_output_image, device_Linear3_output_image, batchsize * Linear3_output_height * sizeof(float), cudaMemcpyDeviceToHost));

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
    CHECK(cudaFree(device_Linear1_output_image));
    // Linear2
    CHECK(cudaFree(device_Linear2_kernel_weight));
    CHECK(cudaFree(device_Linear2_kernel_bias));
    CHECK(cudaFree(device_Linear2_output_image));
    // Linear3
    CHECK(cudaFree(device_Linear3_kernel_weight));
    CHECK(cudaFree(device_Linear3_kernel_bias));
    CHECK(cudaFree(device_Linear3_output_image));

    return 0;
}