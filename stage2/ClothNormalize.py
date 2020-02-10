import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn import init
from dataloader_viton import *

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




class ClothNormalizer(nn.Module):
    def __init__(self, nc=2, ndf=64, depth = 5,mode="conv",bias=False):
        super(ClothNormalizer,self).__init__()
        self.layer1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=bias)
        self.layers = [self.layer1]
        for i in range(depth-1):
            self.layers.append(downconv(ndf*two(i),ndf*two(i+1),mode=mode,bias=bias))
        #self.layers.append()
        self.layers = nn.ModuleList(self.layers)
        self.conv = nn.Conv2d(ndf*two(depth-1),1,3,1,1,bias=bias)
        self.Regressor = nn.Sequential(
            nn.Linear(int((INPUT_SIZE[0]*INPUT_SIZE[1])/(2**(2*depth))),32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(32,6))

    def forward2(self,con_mask,tar_mask):
        x = torch.cat((con_mask,tar_mask),1)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = self.conv(x)
        batch,c,h,w = x.shape
        x = x.view(batch,-1)
        abcdefq = self.Regressor(x)
        a = abcdefq[:,0].view(-1,1,1)
        b = abcdefq[:,1].view(-1,1,1)
        c = abcdefq[:,2].view(-1,1,1)
        d = abcdefq[:,3].view(-1,1,1)
        e = abcdefq[:,4].view(-1,1,1)
        f = abcdefq[:,5].view(-1,1,1)
        q = abcdefq[:,6].view(-1,1,1)
        LU = a*c*torch.cos(q)
        LD = b*c*torch.sin(q)
        RU = -a*d*torch.sin(q)
        RD = b*d*torch.cos(q)
        U = torch.cat([LU,RU],axis=2)
        D = torch.cat([LD,RD],axis=2)
        M = torch.cat([U,D],axis=1)
        ef = torch.cat([e,f],axis=1)
        theta = torch.cat([M,ef],axis=2)
        return theta


        #tt = torch.cat([a,-b,c],axis=2)
        #ttt = torch.cat([b,a,c],axis=2)
        #theta = torch.cat([tt,ttt],axis=1)
        #theta = theta.view(-1,2,3)
        #return theta#

        #grid = F.affine_grid(theta, tar_mask.shape)#
        #return F.grid_sample(con_mask , grid)#,padding_mode="border")#,q,torch.mean(tar_mask)

    def forward(self,con_mask,tar_mask):
        x = torch.cat((con_mask,tar_mask),1)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = self.conv(x)
        batch,c,h,w = x.shape
        x = x.view(batch,-1)
        x = self.Regressor(x)
        theta = x.view(-1,2,3)
        return theta
