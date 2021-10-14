import numpy as np
from pathlib import Path
import torch
import os
import random

import torch.optim as optim
import pandas as pd
import torch.nn as nn
import torch


def conv3x3(in_channels, out_channels, stride=1, activation=True):
    if activation:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
    else:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            nn.BatchNorm2d(out_channels)
        ]


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        # Each enc_conv/dec_conv block should look like this:
        # nn.Sequential(
        #     nn.Conv2d(...),
        #     ... (2 or 3 conv layers with relu and batchnorm),
        # )
        self.enc_conv0 = nn.Sequential(
            *conv3x3(4, 64),
            *conv3x3(64, 64)
        )
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)  # 256 -> 128
        self.enc_conv1 = nn.Sequential(
            *conv3x3(64, 128),
            *conv3x3(128, 128)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)  # 128 -> 64
        self.enc_conv2 = nn.Sequential(
            *conv3x3(128, 256),
            *conv3x3(256, 256),
            *conv3x3(256, 256)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)  # 64 -> 32
        self.enc_conv3 = nn.Sequential(
            *conv3x3(256, 512),
            *conv3x3(512, 512),
            *conv3x3(512, 512)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)  # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            *conv3x3(512, 512),
            *conv3x3(512, 512),
            *conv3x3(512, 512)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)  # 16 -> 8

        # decoder (upsampling)
        self.upsample0 = nn.MaxUnpool2d(2, stride=2)  # 16 -> 32
        # https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html
        self.dec_conv0 = nn.Sequential(
            *conv3x3(512, 512),
            *conv3x3(512, 512),
            *conv3x3(512, 512)
        )
        self.upsample1 = nn.MaxUnpool2d(2, stride=2)  # 32 -> 64
        self.dec_conv1 = nn.Sequential(
            *conv3x3(512, 512),
            *conv3x3(512, 512),
            *conv3x3(512, 256)
        )
        self.upsample2 = nn.MaxUnpool2d(2, stride=2)  # 64 -> 128
        self.dec_conv2 = nn.Sequential(
            *conv3x3(256, 256),
            *conv3x3(256, 256),
            *conv3x3(256, 128)
        )
        self.upsample3 = nn.MaxUnpool2d(2, stride=2)  # 128 -> 256
        self.dec_conv3 = nn.Sequential(
            *conv3x3(128, 64),
            *conv3x3(64, 64)
        )

        self.upsample4 = nn.MaxUnpool2d(2, stride=2)  # 128 -> 256
        self.dec_conv4 = nn.Sequential(
            *conv3x3(64, 64, activation=False),
            *conv3x3(64, 1, activation=False)  # change it no activation function is needed
        )

    def forward(self, x):
        # encoder
        e0, indices0 = self.pool0(self.enc_conv0(x))

        e1, indices1 = self.pool1(self.enc_conv1(e0))
        e2, indices2 = self.pool2(self.enc_conv2(e1))
        e3, indices3 = self.pool3(self.enc_conv3(e2))
        # bottleneck

        b, indices4 = self.pool4(self.bottleneck_conv(e3))
        # decoder
        # print(indices4)

        d0 = self.dec_conv0(self.upsample0(b, indices4, output_size=e3.shape))

        d0 = torch.nn.functional.interpolate(d0, e3.shape[2])
        d1 = self.dec_conv1(self.upsample1(e3 + d0, indices3))
        d2 = self.dec_conv2(self.upsample2(e2 + d1, indices2))
        d3 = self.dec_conv3(self.upsample3(e1 + d2, indices1))
        d4 = self.dec_conv4(self.upsample4(e0 + d3, indices0))  # no activation
        # d4 = F.softmax(d4, 1)
        return d4