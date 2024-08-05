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

std::vector<std::vector<float>> read_mnist_images(const std::string &path) {
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
    std::vector<std::vector<float>> images(num_images, std::vector<float>(image_size));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char *)&pixel, sizeof(pixel));
            images[i][j] = static_cast<float>(pixel) / 255.0f;
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

__global__ void Conv_ReLu(float *input_image, float *output_image, float *kernel_weights, float *kernel_bias, int input_height, int kernel_height, int output_height, int input_channel) {
    float value = 0;
    for (int z = 0; z < input_channel; z++) {
        for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_height; j++) {
                value += input_image[z * input_height * input_height + OFFSET(threadIdx.y + i, threadIdx.x + j, input_height)] * kernel_weights[blockIdx.x * input_channel * kernel_height * kernel_height + z * kernel_height * kernel_height + OFFSET(i, j, kernel_height)];
            }
        }
    }
    value += kernel_bias[blockIdx.x];
    value = value > 0 ? value : 0;
    output_image[blockIdx.x * output_height * output_height + OFFSET(threadIdx.y, threadIdx.x, output_height)] = value;
}

__global__ void MaxPool(float *input_image, float *output_image, int input_height, int output_height) {
    float value = 0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int input_image_index = blockIdx.x * input_height * input_height + OFFSET(threadIdx.y * 2 + i, threadIdx.x * 2 + j, input_height);
            if (input_image[input_image_index] > value) {
                value = input_image[input_image_index];
            }
        }
    }
    output_image[blockIdx.x * output_height * output_height + threadIdx.y * output_height + threadIdx.x] = value;
}

__global__ void Linear_ReLu(float *input_image, float *output_image, float *kernel_weights, float *kernel_bias, int input_height, bool ReLu) {
    float value = 0;
    for (int i = 0; i < input_height; i++) {
        value += input_image[i] * kernel_weights[OFFSET(blockIdx.x, i, input_height)];
    }
    value += kernel_bias[blockIdx.x];
    value = ((!ReLu) || (value > 0)) ? value : 0;
    output_image[blockIdx.x] = value;
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
    const int Conv1_input_height = 28;
    const int Conv1_kernel_height = 5;
    const int Conv1_output_height = Conv1_input_height - Conv1_kernel_height + 1;
    const int Conv1_input_channel = 1;
    const int Conv1_output_channel = 6;
    float *device_input_image;
    float *device_Conv1_output_image;
    float *device_Conv1_kernel_weight;
    float *device_Conv1_kernel_bias;
    CHECK(cudaMalloc((void **)&device_input_image, Conv1_input_channel * Conv1_input_height * Conv1_input_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Conv1_output_image, Conv1_output_channel * Conv1_output_height * Conv1_output_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Conv1_kernel_weight, Conv1_output_channel * Conv1_input_channel * Conv1_kernel_height * Conv1_kernel_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Conv1_kernel_bias, Conv1_output_channel * sizeof(float)));
    CHECK(cudaMemcpy(device_Conv1_kernel_weight, &Conv1_weight[0], Conv1_output_channel * Conv1_input_channel * Conv1_kernel_height * Conv1_kernel_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Conv1_kernel_bias, &Conv1_bias[0], Conv1_output_channel * sizeof(float), cudaMemcpyHostToDevice));
    dim3 Conv1_blocksperGrid(Conv1_output_channel);
    dim3 Conv1_threadsperBlock(Conv1_output_height, Conv1_output_height);

    // MaxPool1 [6, 24, 24] -> [6, 12, 12]
    const int MaxPool1_input_height = Conv1_output_height;
    const int MaxPool1_output_height = MaxPool1_input_height / 2;
    const int MaxPool1_channel = Conv1_output_channel;
    float *device_MaxPool1_output_image;
    CHECK(cudaMalloc((void **)&device_MaxPool1_output_image, MaxPool1_channel * MaxPool1_output_height * MaxPool1_output_height * sizeof(float)));
    dim3 MaxPool1_blocksperGrid(MaxPool1_channel);
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
    CHECK(cudaMalloc((void **)&device_Conv2_output_image, Conv2_output_channel * Conv2_output_height * Conv2_output_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Conv2_kernel_weight, Conv2_output_channel * Conv2_input_channel * Conv2_kernel_height * Conv2_kernel_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Conv2_kernel_bias, Conv2_output_channel * sizeof(float)));
    CHECK(cudaMemcpy(device_Conv2_kernel_weight, &Conv2_weight[0], Conv2_output_channel * Conv2_input_channel * Conv2_kernel_height * Conv2_kernel_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Conv2_kernel_bias, &Conv2_bias[0], Conv2_output_channel * sizeof(float), cudaMemcpyHostToDevice));
    dim3 Conv2_blocksperGrid(Conv2_output_channel);
    dim3 Conv2_threadsperBlock(Conv2_output_height, Conv2_output_height);

    // MaxPool2 [16, 8, 8] -> [16, 4, 4]
    const int MaxPool2_input_height = Conv2_output_height;
    const int MaxPool2_output_height = MaxPool2_input_height / 2;
    const int MaxPool2_channel = Conv2_output_channel;
    float *device_MaxPool2_output_image;
    CHECK(cudaMalloc((void **)&device_MaxPool2_output_image, MaxPool2_channel * MaxPool2_output_height * MaxPool2_output_height * sizeof(float)));
    dim3 MaxPool2_blocksperGrid(MaxPool2_channel);
    dim3 MaxPool2_threadsperBlock(MaxPool2_output_height, MaxPool2_output_height);

    // Linear1 [256] -> [120]
    const int Linear1_input_height = MaxPool2_channel * MaxPool2_output_height * MaxPool2_output_height;
    const int Linear1_output_height = 120;
    float *device_Linear1_output_image;
    float *device_Linear1_kernel_weight;
    float *device_Linear1_kernel_bias;
    CHECK(cudaMalloc((void **)&device_Linear1_output_image, Linear1_output_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear1_kernel_weight, Linear1_output_height * Linear1_input_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear1_kernel_bias, Linear1_output_height * sizeof(float)));
    CHECK(cudaMemcpy(device_Linear1_kernel_weight, &Linear1_weight[0], Linear1_output_height * Linear1_input_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Linear1_kernel_bias, &Linear1_bias[0], Linear1_output_height * sizeof(float), cudaMemcpyHostToDevice));
    dim3 Linear1_blocksperGrid(Linear1_output_height);
    dim3 Linear1_threadsperBlock(1);

    // Linear2 [120] -> [84]
    const int Linear2_input_height = Linear1_output_height;
    const int Linear2_output_height = 84;
    float *device_Linear2_output_image;
    float *device_Linear2_kernel_weight;
    float *device_Linear2_kernel_bias;
    CHECK(cudaMalloc((void **)&device_Linear2_output_image, Linear2_output_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear2_kernel_weight, Linear2_output_height * Linear2_input_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear2_kernel_bias, Linear2_output_height * sizeof(float)));
    CHECK(cudaMemcpy(device_Linear2_kernel_weight, &Linear2_weight[0], Linear2_output_height * Linear2_input_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Linear2_kernel_bias, &Linear2_bias[0], Linear2_output_height * sizeof(float), cudaMemcpyHostToDevice));
    dim3 Linear2_blocksperGrid(Linear2_output_height);
    dim3 Linear2_threadsperBlock(1);

    // Linear3 [84] -> [10]
    const int Linear3_input_height = Linear2_output_height;
    const int Linear3_output_height = 10;
    float *device_Linear3_output_image;
    float *device_Linear3_kernel_weight;
    float *device_Linear3_kernel_bias;
    float *host_Linear3_output_image;
    CHECK(cudaMalloc((void **)&device_Linear3_output_image, Linear3_output_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear3_kernel_weight, Linear3_output_height * Linear3_input_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_Linear3_kernel_bias, Linear3_output_height * sizeof(float)));
    CHECK(cudaMemcpy(device_Linear3_kernel_weight, &Linear3_weight[0], Linear3_output_height * Linear3_input_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_Linear3_kernel_bias, &Linear3_bias[0], Linear3_output_height * sizeof(float), cudaMemcpyHostToDevice));
    host_Linear3_output_image = (float *)malloc(Linear3_output_height * sizeof(float));
    dim3 Linear3_blocksperGrid(Linear3_output_height);
    dim3 Linear3_threadsperBlock(1);

    int correct_nums = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < images.size(); t++) {
        CHECK(cudaMemcpy(device_input_image, &images[t][0], Conv1_input_channel * Conv1_input_height * Conv1_input_height * sizeof(float), cudaMemcpyHostToDevice));
        // Conv1
        Conv_ReLu<<<Conv1_blocksperGrid, Conv1_threadsperBlock>>>(device_input_image, device_Conv1_output_image, device_Conv1_kernel_weight, device_Conv1_kernel_bias, Conv1_input_height, Conv1_kernel_height, Conv1_output_height, Conv1_input_channel);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        // MaxPool1
        MaxPool<<<MaxPool1_blocksperGrid, MaxPool1_threadsperBlock>>>(device_Conv1_output_image, device_MaxPool1_output_image, MaxPool1_input_height, MaxPool1_output_height);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        // Conv2
        Conv_ReLu<<<Conv2_blocksperGrid, Conv2_threadsperBlock>>>(device_MaxPool1_output_image, device_Conv2_output_image, device_Conv2_kernel_weight, device_Conv2_kernel_bias, Conv2_input_height, Conv2_kernel_height, Conv2_output_height, Conv2_input_channel);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        // MaxPool2
        MaxPool<<<MaxPool2_blocksperGrid, MaxPool2_threadsperBlock>>>(device_Conv2_output_image, device_MaxPool2_output_image, MaxPool2_input_height, MaxPool2_output_height);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        // Linear1
        Linear_ReLu<<<Linear1_blocksperGrid, Linear1_threadsperBlock>>>(device_MaxPool2_output_image, device_Linear1_output_image, device_Linear1_kernel_weight, device_Linear1_kernel_bias, Linear1_input_height, true);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        // Linear2
        Linear_ReLu<<<Linear2_blocksperGrid, Linear2_threadsperBlock>>>(device_Linear1_output_image, device_Linear2_output_image, device_Linear2_kernel_weight, device_Linear2_kernel_bias, Linear2_input_height, true);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        // Linear3
        Linear_ReLu<<<Linear3_blocksperGrid, Linear3_threadsperBlock>>>(device_Linear2_output_image, device_Linear3_output_image, device_Linear3_kernel_weight, device_Linear3_kernel_bias, Linear3_input_height, false);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(host_Linear3_output_image, device_Linear3_output_image, Linear3_output_height * sizeof(float), cudaMemcpyDeviceToHost));

        int index = 0;
        for (int k = 0; k < 10; k++) {
            if (host_Linear3_output_image[k] > host_Linear3_output_image[index]) {
                index = k;
            }
        }
        if (index == labels[t]) {
            correct_nums++;
        }
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << float(correct_nums) / float(images.size()) << std::endl;

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