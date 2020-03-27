from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.nn import init
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torchvision import models, transforms
from Models.loss import *

DEBUG = False
MAX_CH = 256
SMOOTH = False 

"""
Two feature pyramid networks - source FPN, target FPN
N encoding layers => downsample conv with stride 2 followed by one residual block
"""

def conv(in_channels, out_channels, stride, kernel_size=3, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size,
		          stride, padding, dilation, bias=bias),
		norm_layer(out_channels),
		)


def deconv(in_channels, out_channels, kernel_size=4, padding=1, activation="relu"):
	if activation == "leaky":
		return nn.Sequential(
				nn.ConvTranspose2d(in_channels, out_channels,
				                   kernel_size=kernel_size, stride=2, padding=padding, bias=True),
				nn.LeakyReLU(0.1, inplace=True)
				)
	else:
		return nn.Sequential(
				nn.ConvTranspose2d(in_channels, out_channels,
				                   kernel_size=kernel_size, stride=2, padding=padding, bias=True),
				nn.ReLU()
				)


def upconv(in_channels, out_channels, mode="transpose"):
	if mode == "transpose":
		return nn.ConvTranspose2d(
			in_channels, out_channels, kernel_size=2, stride=2
			)
	else:
		return nn.Sequential(
			nn.Upsample(mode="bilinear", scale_factor=2),
			nn.Conv2d(in_channels, out_channels, kernel_size=3,
			          stride=1, padding=1, bias=True, groups=1)
			)


def predict_flow(in_channels):
	return nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)


class BasicBlock(nn.Module):
	"""
	One Encoding Layer
	downsample + residual block
	"""

	def __init__(self, in_channels, out_channels, norm="batch", activation="relu"):
		super(BasicBlock, self).__init__()

		if norm == "batch":
			norm_layer = nn.BatchNorm2d
			use_bias = False
		elif norm == "instance":
			norm_layer = nn.InstanceNorm2d
			use_bias = True
		else:
			raise NotImplementedError()
			
		self.conv_block1 = conv(in_channels, out_channels, 2, bias=use_bias, norm_layer=norm_layer)
		self.conv_block2 = conv(out_channels, out_channels, 1, bias=use_bias, norm_layer=norm_layer)

		if activation == "leaky":
			self.activation = nn.LeakyReLU(0.1, inplace=True)
		else:
			self.activation = nn.ReLU(True)

	def forward(self, x):
		residual = self.activation(self.conv_block1(x))
		x = self.activation(self.conv_block2(residual))
		x = self.conv_block2(x)
		return self.activation(x+residual)


"""
class for Spatial Transformer Network
"""
class STN(nn.Module):
	# NEED to check channel number ==> output should be 3*2
	def __init__(self, input_ch):
		super(STN, self).__init__()
	
	def forward(self, x, flow):
		flow = flow.transpose(1,2).transpose(2,3)
		x = F.grid_sample(x, flow, padding_mode="border")
		return x


"""
class for Feature Pyramid Networks
"""
class FPN(nn.Module):
	def __init__(self, N, start_ch, init_ch = 64):
		super(FPN, self).__init__()
		self.N = N

		# encoding layer - left to right
		self.conv = []
		for i in range(self.N):
			if i==0:
				self.conv.append(BasicBlock(start_ch, init_ch))
			else:
				self.conv.append(BasicBlock(init_ch * (2**(i-1)), init_ch * (2**i)))

		# decoding layer - left to right
		self.deconv = []
		for i in range(self.N):
			self.deconv.insert(0, deconv(init_ch * 2**(self.N-i-1), 
				MAX_CH, kernel_size=4, padding=1))

		self.conv = nn.ModuleList(self.conv)
		self.deconv = nn.ModuleList(self.deconv)
		
		# smoothing layer - reduce upsampling aliasing effect
		if SMOOTH:
			self.smooth = nn.Conv2d(MAX_CH, MAX_CH, kernel_size=3, stride=1, padding=1)

	def _upsample_add(self, x, y):
		_, _, H, W = y.size()
		if x is None:
			return y
		else:
			return F.upsample(x, size=(H, W), mode="bilinear") + y

	def forward(self, input):
		encoder = []
		decoder = []

		for i in range(self.N):
			encoder.append(self.conv[i](input if i==0 else encoder[-1]))
			if DEBUG:
				print('shape of encoder is {}'.format(encoder[-1].shape))

		for i in range(self.N):
			x = self._upsample_add(None if i==0 else decoder[0], self.deconv[-i-1](encoder[-i-1]))
			decoder.insert(0, x)
			if DEBUG:
				print('shape of decoder is {}'.format(decoder[0].shape))

		if SMOOTH:
			_smooth = []
			_smooth.append(decoder[0])
			for i in range(self.N-1):
				_smooth.append(self.smooth(decoder[i+1]))
			return _smooth
		else:
			return decoder

class FlowNet(nn.Module):
	def __init__(self, N=4, src_ch=4, tar_ch=1):
		super(FlowNet, self).__init__()
		self.N = N

		self.SourceFPN = FPN(self.N, src_ch)
		self.TargetFPN = FPN(self.N, tar_ch)

		# list for Warp - left to right
		L = STN(2)
		init_weights(L, 'xavier')
		self.stn = L
	

		# E layer - left to right
		self.E = []
		for i in range(self.N):
			L = predict_flow(MAX_CH * 2)
			init_weights(L, 'xavier')
			self.E.append(L)

		self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
		self.E = nn.ModuleList(self.E)

	# def get_N(self):
	# 	return self.N

	def forward(self, src, tar):
			src_conv = self.SourceFPN(src)
			tar_conv = self.TargetFPN(tar)
			
			# concat for E4 to E1
			self.F = []
			self.F.append(self.E[-1](torch.cat([src_conv[-1], tar_conv[-1]], 1))) #[W, H, 2]
			
			for i in range(self.N - 1):
				upsample_F = self.upsample(self.F[0]) #[2W, 2H, 2]
				warp = self.stn(src_conv[-i-2], upsample_F)  
				concat = torch.cat([warp, tar_conv[-i-2]], 1)
				self.F.insert(0, upsample_F.add(self.E[-i-1](concat)))

			self.result = self.stn(src, self.F[0])
			self.warp_cloth = self.stn(src[:,0:3,:,:], self.F[0])
			self.warp_mask = self.stn(src[:,3:4,:,:], self.F[0])
			self.tar_mask = tar

			return self.F, self.warp_cloth, self.warp_mask

def test_FPN():
	return FPN(4, [3, 32, 64, 128])

def test_STN():
	return STN(256)

def test_FlowNet():
	return FlowNet(5, 4, 1)

def test():
	net = test_FlowNet()
	fms = net(Variable(torch.randn(1, 4, 192, 256)), Variable(torch.randn(1, 1, 192, 256)))


###################################
######## Weight initialize ########
###################################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


if __name__ == "__main__":
	f = FlowNet(5,4,1)
	print(f)
