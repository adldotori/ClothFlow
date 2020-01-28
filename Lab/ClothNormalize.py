import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn import init

def two(n,maxtwo=512):
    return min(2**n,maxtwo)

class downconv(nn.Module):
    def __init__(self,in_channel,out_channel,mode,bias):
        super(downconv,self).__init__()
        if mode == "conv":
            self.module = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=bias),
                nn.InstanceNorm2d(out_channel, affine=True),
                nn.LeakyReLU(0.2,inplace=True)
            )
        if mode == "pool":
            self.module = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=7,padding=3,bias=bias),
                nn.MaxPool2d(2, stride=2),
                nn.InstanceNorm2d(out_channel, affine=True),
                nn.LeakyReLU(0.2,inplace=True)
            )
    def forward(self,x):
        return self.module(x)

class 


class ClothNormalizer(nn.Module):
    def __init__(self, nc=2, ndf=64, depth = 5,mode="conv",bias=False):
        super(Encoder,self).__init__()
        self.layer1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=bias)
        self.layers = [layer1]
        for i in range(depth-1):
            self.layers.append(downconv(ndf*two(i),ndf*two(i+1),mode=mode,bias=bias))
        self.layers.append()
        self.layers = nn.ModuleList(self.layers)
        self.conv = nn.Conv2d(ndf*two(depth-1),1,3,1,1,bias=bias)
        self.Regressor = nn.Sequential(
            nn.Linear(192*256/(2**(2*depth)),32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(32,6))
        

    def forward(self,con_mask,tar_mask):
        x = torch.cat((con_mask,tar_mask),1)
        for i in range(len(layers)):
            x = self.layers[i](x)
        theta = self.Regressor(x)
        theta = theta.view(-1,2,3)
        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(con_mask, grid)

