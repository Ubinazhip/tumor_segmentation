import numpy as np
import torch.optim as optim
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from time import time

def conv3x3(in_channels, out_channels, stride=1,kernel_size = (3,3), activation = True):
    if activation:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
    else:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(1, 1)),
            nn.BatchNorm2d(out_channels)
        ]

def last_conv(in_channels, out_channels, stride = 1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride),
    ]

class UNet(nn.Module):
    def __init__(self, out = 1):
        super().__init__()

        self.enc_conv0 = nn.Sequential(
            *conv3x3(4, 64),
            *conv3x3(64, 64)
        )
        self.pool0 =  nn.MaxPool2d(2, 2) # 240->120
        self.enc_conv1 = nn.Sequential(
            *conv3x3(64, 128),
            *conv3x3(128, 128)
        )
        self.pool1 =  nn.MaxPool2d(2, 2)# 120 -> 60
        self.enc_conv2 =  nn.Sequential(
            *conv3x3(128, 256),
            *conv3x3(256, 256)
        )
        self.pool2 =  nn.MaxPool2d(2, 2)# 120 -> 60
        self.enc_conv3 = nn.Sequential(
            *conv3x3(256, 512),
            *conv3x3(512, 512)
        )
        self.pool3 =  nn.MaxPool2d(2, 2)# 60->30

        self.bottleneck_conv = nn.Sequential(
            *conv3x3(512, 1024),
            *conv3x3(1024, 1024) #30->15
        )

        self.upsample0 =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(1024, 512, kernel_size = (2, 2), padding = (0, 0)))# 15 -> 29
        self.dec_conv0 = nn.Sequential(
            *conv3x3(1024, 512),
            *conv3x3(512, 512)
        ) 
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),# 16 -> 32
            nn.Conv2d(512, 256, kernel_size = (2, 2), padding = (1, 1)))# 29 -> 59
        self.dec_conv1 = nn.Sequential(
            *conv3x3(512, 256),
            *conv3x3(256, 256)
        )
        self.upsample2 =   nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),# 16 -> 32
            nn.Conv2d(256, 128, kernel_size = (2, 2), padding = (1, 1)))# 59 -> 119
        
        self.dec_conv2 = nn.Sequential(
            *conv3x3(256, 128),
            *conv3x3(128, 128)
        )
        self.upsample3 =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),# 16 -> 32
            nn.Conv2d(128, 64, kernel_size = (2, 2), padding = (1, 1))) # 119 -> 239
        self.dec_conv3 =   nn.Sequential(
            *conv3x3(128, 64, kernel_size = 2, activation = False),
            *conv3x3(64, 64, activation = False),
            *last_conv(64, out)
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

       # d3 = torch.sigmoid(d3)
        return d3