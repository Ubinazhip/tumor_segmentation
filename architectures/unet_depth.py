import numpy as np
import torch.optim as optim
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from time import time

def depthwise_seperable_conv(nin, nout, kernels_per_layer = 1, kernel_size = 3, padding = 1, stride = 1, activation = False):
    if activation == False:
        return [
            nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, stride = stride,groups=nin),
            nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1),
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace = True)]
    else:
        return 
        [
            nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, stride = stride,groups=nin),
            nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1),
           
        ]


def last_conv(in_channels, out_channels, stride = 1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride),
    ]

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc_conv0 = nn.Sequential(
            *depthwise_seperable_conv(4, 64),
            *depthwise_seperable_conv(64, 64)
            
        )
        self.pool0 =  nn.MaxPool2d(2, 2) # 240->120
        self.enc_conv1 = nn.Sequential(
            *depthwise_seperable_conv(64, 128),
            *depthwise_seperable_conv(128, 128)

        )
        self.pool1 =  nn.MaxPool2d(2, 2)# 120 -> 60
        self.enc_conv2 =  nn.Sequential(
            *depthwise_seperable_conv(128, 256),
            *depthwise_seperable_conv(256, 256)

        )
        self.pool2 =  nn.MaxPool2d(2, 2)# 120 -> 60
        self.enc_conv3 = nn.Sequential(
            *depthwise_seperable_conv(256, 512),
            *depthwise_seperable_conv(512, 512)

        )
        self.pool3 =  nn.MaxPool2d(2, 2)# 60->30

        self.bottleneck_conv = nn.Sequential(
            *depthwise_seperable_conv(512, 1024),
            *depthwise_seperable_conv(1024, 1024)
        )

        self.upsample0 =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            *depthwise_seperable_conv(1024, 512, kernel_size = 2, padding = 0))
 
        self.dec_conv0 = nn.Sequential(
            *depthwise_seperable_conv(1024, 512),
            *depthwise_seperable_conv(512, 512)

        ) 
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),# 16 -> 32
            *depthwise_seperable_conv(512, 256, kernel_size = 2, padding = 1))

        self.dec_conv1 = nn.Sequential(
            *depthwise_seperable_conv(512, 256),
            *depthwise_seperable_conv(256, 256)

        )
        self.upsample2 =   nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),# 16 -> 32
            *depthwise_seperable_conv(256, 128, kernel_size = 2, padding = 1))

        
        self.dec_conv2 = nn.Sequential(
            *depthwise_seperable_conv(256, 128),
            *depthwise_seperable_conv(128, 128)

        )
        self.upsample3 =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),# 16 -> 32
            *depthwise_seperable_conv(128, 64, kernel_size = 2, padding = 1))

        self.dec_conv3 =   nn.Sequential(
            *depthwise_seperable_conv(128, 64, kernel_size = 2, activation = False),
            *depthwise_seperable_conv(64, 64, activation = False),

            *last_conv(64, 1)
        )

    def forward(self, x):
      #  print(f'input size is {x.shape}')
        e0 = self.enc_conv0(x)
        
        pool_e0 = self.pool0(e0)
        e1 = self.enc_conv1(pool_e0)
        pool_e1 = self.pool1(e1)
        e2 = self.enc_conv2(pool_e1)
        pool_e2 = self.pool2(e2)
        e3 = self.enc_conv3(pool_e2)
        pool_e3 = self.pool3(e3)
 
        
        b = self.bottleneck_conv(pool_e3)
        upsampled0 = self.upsample0(b)
        interpolated0 = torch.nn.functional.interpolate(e3, upsampled0.shape[2])
        d0 = self.dec_conv0(torch.cat((upsampled0, interpolated0), 1))
        upsampled1 = self.upsample1(d0)
        interpolated1 = torch.nn.functional.interpolate(e2, upsampled1.shape[2])
        d1 = self.dec_conv1(torch.cat((upsampled1, interpolated1), 1))
                
        upsampled2 = self.upsample2(d1)
        interpolated2 = torch.nn.functional.interpolate(e1, upsampled2.shape[2])
        d2 = self.dec_conv2(torch.cat((upsampled2, interpolated2), 1))
        
        upsampled3 = self.upsample3(d2)
        interpolated3 = torch.nn.functional.interpolate(e0, upsampled3.shape[2])
        d3 = self.dec_conv3(torch.cat((upsampled3, interpolated3), 1))
        #d3 = torch.nn.functional.interpolate(d3, x.shape[2])
        #d3 = F.softmax(d3, 1)
        
        return d3