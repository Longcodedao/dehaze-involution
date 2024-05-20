import torch
import torch.nn as nn
import torch.nn.functional as F
from .involution import Involution_CUDA, Involution

import torchvision.transforms.functional as TF

class UNET_block(nn.Module):
    def __init__(self, in_channels, out_channels, use_cuda = True):
        super(UNET_block,self).__init__()

        self.involution_block = Involution_CUDA if use_cuda else Involution
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
            self.involution_block(out_channels, kernel_size = 7, stride = 1),
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


def unet_block(in_channels, out_channels):
  return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 3, 1, 1),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, 3, 1, 1),
      nn.ReLU()
  )


class UNET(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, 
                 features = [64, 128, 256, 512], device = 'cpu', use_cuda = True):
        super().__init__()
        self.in_channels = in_channels
        in_channels_temp = in_channels
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        self.final_layer = nn.Conv2d(in_channels = features[0],
                                     out_channels = out_channels,
                                     kernel_size = 1)
        
        self.bottle_neck = UNET_block(in_channels = features[-1],
                                      out_channels = features[-1] * 2,
                                     use_cuda = use_cuda)
        
        # down
        for feature in features:
            self.downs.append(
                UNET_block(in_channels_temp,
                           out_channels = feature,
                          use_cuda = use_cuda))
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
                                       out_channels = feature,
                                      use_cuda = use_cuda))

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
    


class Unet(nn.Module):
  def __init__(self, n_channels):
    super().__init__()
    self.n_channels = n_channels
    self.downsample = nn.MaxPool2d(2)
    self.upsample = nn.Upsample(scale_factor = 2, mode ="bilinear") # Phep noi suy
    self.block_down1 = unet_block(3, 64)
    self.block_down2 = unet_block(64, 128)
    self.block_down3 = unet_block(128, 256)

    self.block_down4 = unet_block (256, 512)
    self.block_neck = unet_block (512, 1024)

    self.block_up1 = unet_block(1024+ 512, 512)
    self.block_up2 = unet_block( 512+ 256, 256)
    self.block_up3 = unet_block(128 + 256, 128)
    self.block_up4 = unet_block(128 + 64, 64)
    self.conv_cls = nn.Conv2d(64, n_channels, 1) # ( B, n_classes, H, W)

  def forward(self, x):
    #( B, C, H , W)
    x1 = self.block_down1(x)
    x =self.downsample(x1)
    x2 = self.block_down2(x)
    x = self.downsample(x2)
    x3 = self.block_down3(x)
    x = self.downsample(x3)
    x4 = self.block_down4(x)
    x = self.downsample(x4)

    x = self.block_neck(x)


# (B, 1024, H, W) cat (B, 512, H, W) = (B, 1536, H, w)

    x = torch.cat([x4, self.upsample(x)], dim = 1)
    x = self.block_up1(x)
    x = torch.cat([x3, self.upsample(x)], dim = 1)
    x = self.block_up2(x)
    x = torch.cat([x2, self.upsample(x)], dim = 1)
    x = self.block_up3(x)
    x = torch.cat([x1, self.upsample(x)], dim = 1)
    x = self.block_up4(x)

    x = self.conv_cls(x)

    return x