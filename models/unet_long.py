import torch
import torch.nn as nn
import torch.nn.functional as F
from .involution import Involution_CUDA

import torchvision.transforms.functional as TF



class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Normalize across the channel dimension
        return F.normalize(x, dim=1) * self.scale * self.gamma.view(1, -1, 1, 1)
    
    
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(LayerNorm2d, self).__init__()
        self.layer_norm = nn.LayerNorm([num_channels, 1, 1], eps=eps)

    def forward(self, x):
        # Reshape to (batch_size, num_channels, height * width)
        batch_size, num_channels, height, width = x.size()
        x_reshaped = x.view(batch_size, num_channels, -1)
        
        # Apply LayerNorm and then reshape back
        x_norm = self.layer_norm(x_reshaped)
        x_norm = x_norm.view(batch_size, num_channels, height, width)
        
        return x_norm

    

class BottleNeck_Long(nn.Module):
    def __init__(self, in_channels, out_channels, inv_kernel = 7):
        super(BottleNeck_Long, self).__init__()
        self.invblock = nn.Sequential(
            Involution_CUDA(in_channels, kernel_size = inv_kernel, stride = 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,
                      bias = False),
            nn.BatchNorm2d(out_channels),
        )   

        self.mapping=nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      padding = 0),
            nn.BatchNorm2d(out_channels))
        
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.invblock(x)
        x1 = self.conv_block(x1)

        if x.shape[1] != x1.shape[1]:
            x = self.mapping(x)

        return self.gelu(x1 + x)


class UNET_Long(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, 
                 features = [64, 128, 256, 512], device = 'cpu'):
        super().__init__()
        self.in_channels = in_channels
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.warmup_conv = nn.Conv2d(in_channels = in_channels, 
                                     out_channels = features[0],
                                     kernel_size = 3,
                                     padding = 1)
        
        self.final_layer = nn.Conv2d(in_channels = features[0],
                                     out_channels = out_channels,
                                     kernel_size = 1)
        
        self.bottle_neck = BottleNeck_Long(in_channels = features[-1],
                                      out_channels = features[-1] * 2)

        in_channels_temp = features[0]

        # down
        for i in range(len(features)):
            in_channels_temp = features[i]
            self.downs.append(
                BottleNeck_Long(in_channels_temp,
                           out_channels = in_channels_temp))
            
            if i < len(features) - 1:
                # Instead of using AvgPool2D
                self.downs.append(nn.Conv2d(in_channels = in_channels_temp,
                                        out_channels = features[i + 1],
                                        kernel_size = 1,
                                        stride = 2,
                                        padding = 1))
                


        # ups
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(
                in_channels = feature * 2,
                out_channels = feature, 
                kernel_size = 2, stride = 2))
            self.ups.append(BottleNeck_Long(in_channels = feature * 2, 
                                       out_channels = feature))

        self.device = device

        # print(self.downs)
        
    def forward(self,x):
        skip_connections=[]
        print("Encoder")
        x = self.warmup_conv(x)
        print(x.shape)

    
        for i in range(0, len(self.downs)):
            x = self.downs[i](x)

            if i % 2 == 0:
                skip_connections.append(x)
                print("Involution Block:", x.shape)
            else:
                print("Downsample:", x.shape)


            
        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]
        print(f"After Bottleneck: ", x.shape)

        print("Decoder")
        for i in range(0, len(self.ups), 2):

            x = self.ups[i](x)
            print(f"Upsample: ", x.shape)
            skip_connection = skip_connections[i//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, skip_connection.shape[2::], antialias=True)

            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups[i+1](concat_skip)
            print(f"Involution Block: ", x.shape)

        return self.final_layer(x)
