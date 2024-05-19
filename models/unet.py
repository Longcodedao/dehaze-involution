import torch
import torch.nn as nn
import torch.nn.functional as F
from .involution import Involution_CUDA

import torchvision.transforms.functional as TF

class UNET_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNET_block,self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,
                      bias = False),
            nn.GELU()

        )
        
        self.involution = nn.Sequential(
            Involution_CUDA(out_channels, kernel_size = 7, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU())
        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels = out_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,
                      bias = False),
            nn.BatchNorm2d(out_channels))
        
        self.gelu = nn.GELU()
        
        self.mapping=nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      padding = 0),
            nn.BatchNorm2d(out_channels))

    
    def forward(self,x):
        x1=self.convblock(x)
        x1=self.involution(x1)
        x1=self.convblock2(x1)

        # For matching the output dimension
        if x.shape[1] != x1.shape[1]:
            x = self.mapping(x)
            
        return self.gelu(x1 + x)



class UNET(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, 
                 features = [64, 128, 256, 512], device = 'cpu'):
        super().__init__()
        self.in_channels = in_channels
        in_channels_temp = in_channels
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        self.final_layer = nn.Conv2d(in_channels = features[0],
                                     out_channels = out_channels,
                                     kernel_size = 1)
        
        self.bottle_neck = UNET_block(in_channels = features[-1],
                                      out_channels = features[-1] * 2)
        
        # down
        for feature in features:
            self.downs.append(
                UNET_block(in_channels_temp,
                           out_channels = feature))
            in_channels_temp = feature
            
            # Instead of using AvgPool2D
            self.downs.append(nn.Conv2d(in_channels = in_channels_temp,
                                     out_channels = in_channels_temp,
                                     kernel_size = 1,
                                     stride = 2,
                                     padding = 0))


        # ups
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(
                in_channels = feature * 2,
                out_channels = feature, 
                kernel_size = 2, stride = 2))
            self.ups.append(UNET_block(in_channels = feature * 2, 
                                       out_channels = feature))

        self.device = device


    def forward(self,x):
        skip_connections=[]

        for i in range(0, len(self.downs), 2):
            x = self.downs[i](x)
            skip_connections.append(x)
            x = self.downs[i + 1](x)
            
        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]
        
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, skip_connection.shape[2::], antialias=True)

            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups[i+1](concat_skip)

        return self.final_layer(x)

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed