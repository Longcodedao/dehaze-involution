#include <torch/extension.h>

torch::Tensor involution_kernel_forward(torch::Tensor input_data,
            torch::Tensor weight_data,  int input_height, int input_width, 
            int output_height, int output_width, 
            int groups, int kernel_h, int kernel_w,
            int stride_h, int stride_w, int dilation_h, int dilation_w,
            int pad_h, int pad_w);

torch::Tensor involution_backward_weight(torch::Tensor grad_output, 
    torch::Tensor input_data, int input_height, int input_width, 
    int output_height, int output_width, int groups, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int dilation_h, int dilation_w,
    int pad_h, int pad_w
);

torch::Tensor involution_backward_input(torch::Tensor grad_output, 
    torch::Tensor weight_data, int input_height, int input_width, 
    int output_height, int output_width, int groups, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int dilation_h, int dilation_w,
    int pad_h, int pad_w
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("involution_kernel_forward", &involution_kernel_forward, "Forward pass for the involution kernel",
        py::arg("input_data"), py::arg("weight_data"), py::arg("input_height"),
        py::arg("input_width"), py::arg("output_height"), py::arg("output_width"),
        py::arg("groups"), py::arg("kernel_h"), py::arg("kernel_w"),
        py::arg("stride_h"), py::arg("stride_w"), py::arg("dilation_h"),
        py::arg("dilation_w"), py::arg("pad_h"), py::arg("pad_w"));

    m.def("involution_backward_weight", &involution_backward_weight, "Backward pass for weights in the involution kernel",
        py::arg("grad_output"), py::arg("input_data"), py::arg("input_height"),
        py::arg("input_width"), py::arg("output_height"), py::arg("output_width"),
        py::arg("groups"), py::arg("kernel_h"), py::arg("kernel_w"),
        py::arg("stride_h"), py::arg("stride_w"), py::arg("dilation_h"),
        py::arg("dilation_w"), py::arg("pad_h"), py::arg("pad_w"));

    m.def("involution_backward_input", &involution_backward_input, "Backward pass for input in the involution kernel",
        py::arg("grad_output"), py::arg("weight_data"), py::arg("input_height"),
        py::arg("input_width"), py::arg("output_height"), py::arg("output_width"),
        py::arg("groups"), py::arg("kernel_h"), py::arg("kernel_w"),
        py::arg("stride_h"), py::arg("stride_w"), py::arg("dilation_h"),
        py::arg("dilation_w"), py::arg("pad_h"), py::arg("pad_w"));
}
