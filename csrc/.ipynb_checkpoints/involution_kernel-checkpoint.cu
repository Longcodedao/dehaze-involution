#include <cuda_runtime.h>
#include <stdexcept>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
#include <bits/stdc++.h>

using namespace std;
#define uint unsigned int

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


__global__ void involution_forward(float *output_data, const float *input_data, 
    const float *weight_data, int batch_size, int channels, int groups, 
    int output_height, int output_width, int input_height, int input_width,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
    int pad_h, int pad_w){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < (batch_size * output_height * output_width * channels)){
        int b = index / (channels * output_height * output_width);
        int c = (index / (output_height * output_width)) % channels;
        int h = (index / (output_width)) % output_height;
        int w = index % output_width;
        int g = c / (channels / groups);

        float value = 0;

        #pragma unroll
        for (int kh = 0; kh < kernel_h; kh++){

            #pragma unroll
            for (int kw = 0; kw < kernel_w; kw++){
                int h_in = -pad_h + h * stride_h + kh * dilation_h;
                int w_in = -pad_w + w * stride_w + kw * dilation_w;

                if ((h_in >= 0) && (h_in < input_height) && (w_in >= 0) && (w_in < input_width)){
                    int offset_pos = ((b * channels + c) * input_height + h_in) * input_width + w_in;

                    // Size of the weight is B, G, K, K, H, W
                    int offset_weight = ((((b * groups + g) * kernel_h + kh) * kernel_w + kw) 
                                            * output_height + h) * output_width + w;

                    value += input_data[offset_pos] * weight_data[offset_weight];
                }
            }
        }

        output_data[index] = value;
    }
}

__global__ void involution_backward_grad_weight(const float *grad_output, const float *input_data,
    float *grad_weight, int batch_size, int channels, int groups, 
    int output_height, int output_width, int input_height, int input_width,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
    int pad_h, int pad_w
) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Shape of the Weight is (B, G, K_h, K_w, o_H, o_W)
    if (index < (batch_size * groups * kernel_h * kernel_w * output_height * output_width)){
        int h = (index / output_height) % output_height;
        int w = index % output_width;
        int kh = (index / kernel_w / output_height / output_width) % kernel_h;
        int kw = (index / output_height / output_width) % kernel_w;
        int h_in = -pad_h + h * stride_h + kh * dilation_h;
        int w_in = -pad_w + w * stride_w + kw * dilation_w;

        if ((h_in >= 0) && (h_in < input_height) && (w_in >= 0) && (w_in < input_width)){
            int g = (index / kernel_h / kernel_w / output_height / output_width) % groups;
            int b = index / groups / kernel_h / kernel_w / output_height / output_width;

            float value = 0;

            #pragma unroll
            for (int c = g * (channels / groups); c < (g + 1) * (channels / groups); c++){
                int offset_grad = ((b * channels + c) * output_height + h) * output_width + w;
                int offset_input = ((b * channels + c) * input_height + h_in) * input_width + w_in;
                
                value += grad_output[offset_grad] * input_data[offset_input];
            }

            grad_weight[index] = value;
        } else {
            grad_weight[index] = 0;
        }
    }

}


__global__ void involution_backward_grad_input(const float *grad_output, const float *weight_data,
    float *grad_input, int batch_size, int channels, int groups,
    int output_height, int output_width, int input_height, int input_width,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int dilation_h,
    int dilation_w, int pad_h, int pad_w
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < (batch_size * channels * input_height * input_width)) {
        int n = index / (channels * input_height * input_width);
        int c = (index / (input_height * input_width)) % channels;
        int h = (index / input_width ) % channels;
        int w = index % input_width;
        int g = c / (channels / groups);

        float value = 0;
        #pragma unroll
        for (int kh = 0; kh < kernel_h; kh++){
            
            #pragma unroll
            for (int kw = 0; kw < kernel_w; kw++){  
                int h_out = pad_h + h - kh * dilation_h;
                int w_out = pad_w + w - kw * dilation_w;

                if ((h_out % stride_h == 0) && (w_out % stride_w == 0)){
                    h_out = h_out / stride_h;
                    w_out = w_out / stride_w;

                    if ((h_out >= 0) &&  (h_out < output_height) 
                        && (w_out >= 0) && (w_out < output_width)) {

                        int offset_output = ((n * channels + c) * output_height + h_out) * output_width + w_out;
                        int offset_weight = ((((n * groups + g) * kernel_h + kh) * kernel_w + kw) 
                                            * output_height + h_out) * output_width + w_out;
                        
                        value += grad_output[offset_output] * weight_data[offset_weight];
                    }
                }

            }
        }

        grad_input[index] = value;
    }

}


inline uint cdiv(uint a, uint b){
    return (a + b - 1) / b;
}



torch::Tensor involution_kernel_forward(torch::Tensor input_data,
            torch::Tensor weight_data, int input_height, int input_width, 
            int output_height, int output_width,
            int groups, int kernel_h, int kernel_w,
            int stride_h, int stride_w, int dilation_h, int dilation_w,
            int pad_h, int pad_w){

    CHECK_CUDA(input_data);
    CHECK_CUDA(weight_data);

    auto device = input_data.device();
    auto batch_size = input_data.size(0);
    auto channels = input_data.size(1);

    auto output_data = torch::zeros({batch_size, channels, output_height, output_width},
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    uint num_elements = (uint) batch_size * channels * output_height * output_width;
    //cout << "Number of elements" << num_elements << endl;
    dim3 num_threads(1024, 1, 1);
    dim3 num_blocks(cdiv(num_elements, 1024), 1, 1);

    //cout << "Number of blocks: " << cdiv(num_elements, 1024) << endl;

    involution_forward<<<num_blocks, num_threads>>>(
        output_data.data_ptr<float>(),
        input_data.data_ptr<float>(),
        weight_data.data_ptr<float>(),
        batch_size, 
        channels, 
        groups, 
        output_height, output_width,
        input_height, input_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        pad_h, pad_w
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output_data;

}



torch::Tensor involution_backward_weight(torch::Tensor grad_output, 
    torch::Tensor input_data, int input_height, int input_width, 
    int output_height, int output_width, int groups, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int dilation_h, int dilation_w,
    int pad_h, int pad_w) { 

    CHECK_INPUT(grad_output);
    CHECK_INPUT(input_data);

    auto device = input_data.device();
    auto batch_size = input_data.size(0);
    auto channels = input_data.size(1);

    uint num_elements = (uint) batch_size * groups * kernel_h * kernel_w * output_height * output_width;

    dim3 num_threads(1024, 1, 1);
    dim3 num_blocks(cdiv(num_elements, 1024), 1, 1);


    auto grad_weight = torch::zeros({batch_size, groups, kernel_h, kernel_w, output_height, output_width},
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    involution_backward_grad_weight<<<num_blocks, num_threads>>>(
        grad_output.data_ptr<float>(),
        input_data.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        batch_size,
        channels,
        groups,
        output_height, output_width,
        input_height, input_width, 
        kernel_h, kernel_w, 
        stride_h, stride_w, 
        dilation_h, dilation_w,
        pad_h, pad_w
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return grad_weight;
}


// const float *grad_output, const float *weight_data,
//     float *grad_input, int batch_size, int channels, int groups,
//     int output_height, int output_width, int input_height, int input_width,
//     int kernel_h, int kernel_w, int stride_h, int stride_w, int dilation_h,
//     int dilation_w, int pad_h, int pad_w


torch::Tensor involution_backward_input(torch::Tensor grad_output, 
    torch::Tensor weight_data, int input_height, int input_width, 
    int output_height, int output_width, int groups, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int dilation_h, int dilation_w,
    int pad_h, int pad_w
) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(weight_data);   

    // The size of weight_data (B, G, K, K, H, W)
    auto device = weight_data.device();
    auto batch_size = weight_data.size(0);
    auto channels = grad_output.size(1);

    uint num_elements = (uint) batch_size * channels * input_height * input_width;

    auto grad_input = torch::zeros({batch_size, channels, input_height, input_width},
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    dim3 num_threads(1024, 1, 1);
    dim3 num_blocks(cdiv(num_elements, 1024), 1, 1);

    involution_backward_grad_input<<<num_blocks, num_threads>>>(
        grad_output.data_ptr<float>(),
        weight_data.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        batch_size, 
        channels, 
        groups,
        output_height, output_width,
        input_height, input_width,
        kernel_h, kernel_w, 
        stride_h, stride_w,
        dilation_h, dilation_w,
        pad_h, pad_w
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return grad_input;
}