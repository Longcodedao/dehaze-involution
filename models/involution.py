import torch
import torch.nn as nn
import involution_cuda 
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 use_bn = True, use_act = True, 
                 stride = 1, padding = 0, dilation = 1):
        super().__init__()

        self.kernel_size = kernel_size
        self.use_bn = use_bn
        self.use_act = use_act

        if self.use_bn:
            self.batch_norm = nn.BatchNorm2d(out_channels)

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                 stride = stride, padding = padding, dilation = dilation)
        
    def forward(self, x):
        x = self.conv2d(x)

        if self.use_bn:
            x = self.batch_norm(x)
        
        if self.use_act:
            x = F.relu(x)

        return x


class _involution(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):

        # We specify the dimension of stride, padding and dilation to 2: x and y
        # Example stride = [stride_h, stride_w]
        assert input.dim() == 4 and input.is_cuda   # (B, C, H, W)
        assert weight.dim() == 6 and weight.is_cuda # (B, G, K, K, H, W) (Output height and width)

        input_height, input_width = input.size(2), input.size(3)
        groups = weight.size(1)
        kernel_h, kernel_w = weight.size(2), weight.size(3)


        output_height = int((input_height + 2 * padding[0] -\
                             (dilation[0] * (kernel_h - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - \
                             (dilation[1] * (kernel_w - 1) + 1)) / stride[1] + 1)
        

        output = involution_cuda.involution_kernel_forward(
            input, weight, input_height, input_width,
            output_height, output_width, groups, kernel_h, 
            kernel_w, stride[0], stride[1], dilation[0], dilation[1],
            padding[0], padding[1]
        )
        
        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation

        return output
    

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda and grad_output.is_contiguous()

        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        input_height, input_width = input.size(2), input.size(3)
        groups = weight.size(1)
        kernel_h, kernel_w = weight.size(2), weight.size(3)

        output_width, output_height = grad_output.size()[2:]

        grad_input, grad_weight = None, None


        if ctx.needs_input_grad[0]:
            grad_input = involution_cuda.involution_backward_input(
                grad_output, weight, input_height, input_width,
                output_height, output_width, groups, kernel_h, 
                kernel_w, stride[0], stride[1], dilation[0], dilation[1],
                padding[0], padding[1]
            )

        if ctx.needs_input_grad[1]:
            grad_weight = involution_cuda.involution_backward_weight(
                grad_output, input, input_height, input_width,
                output_height, output_width, groups, kernel_h, 
                kernel_w, stride[0], stride[1], dilation[0], dilation[1],
                padding[0], padding[1]
            )

        return grad_input, grad_weight, None, None, None


def _involution_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """ involution kernel
    """
    assert input.size(0) == weight.size(0)
    assert input.size(-2)//stride == weight.size(-2)
    assert input.size(-1)//stride == weight.size(-1)

    if input.is_cuda:
        out = _involution.apply(input, weight, _pair(stride), _pair(padding), _pair(dilation))
        if bias is not None:
            out += bias.view(1,-1,1,1)
    else:
        raise NotImplementedError
    return out

class Involution_CUDA(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(Involution_CUDA, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.stride = stride

        reduction_ratio = 4
        self.group_channels = 1
        self.groups = channels // self.group_channels

        self.conv1 = ConvModule(in_channels = channels,
                                out_channels = channels // reduction_ratio,
                                kernel_size = 1)
        
        self.conv2 = ConvModule(in_channels = channels // reduction_ratio,
                                out_channels = kernel_size ** 2 * self.groups,
                                kernel_size = 1,
                                stride = 1,
                                use_act = False,
                                use_bn = False)
        
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, _, h, w = weight.shape

        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)

        pad = (self.kernel_size - 1) // 2
        out = _involution_cuda(x, weight, stride = self.stride, padding = pad)

        return out
    
