from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.autograd import Variable
from torchvision import models

from loss import *

device = torch.device("cuda:2")
DEBUG = True
MAX_CH = 256
"""
Two feature pyramid networks - source FPN, target FPN
N encoding layers => downsample conv with stride 2 followed by one residual block
N = 4 or 5
"""

def conv(in_channels, out_channels, stride, kernel_size=3, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
	model = nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
		norm_layer(out_channels),
		)
	return model

def deconv(in_channels, out_channels, kernel_size=4, padding=1, activation = "relu"):
	if activation == "leaky":
		return nn.Sequential(
				nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=True),
				nn.LeakyReLU(0.1, inplace=True)
				)
	else:
		return nn.Sequential(
				nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=True),
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
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=1)
			)

def predict_flow(in_channels):
	return nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)


class BasicBlock(nn.Module):
	"""
	One Encoding Layer
	downsample + residual block
	"""

	def __init__(self, c_num, i, norm="batch", activation="relu"):
		super(BasicBlock, self).__init__()
		
		if norm == "batch":
			norm_layer = nn.BatchNorm2d
			use_bias = False
		elif norm == "instance":
			norm_layer = nn.InstanceNorm2d
			use_bias = True
		else:
			raise NotImplementedError()

		if (i==0):
			self.conv_block1 = conv(c_num, 64, 2, bias=use_bias, norm_layer=norm_layer)
			self.conv_block2 = conv(64, 64, 1, bias=use_bias, norm_layer=norm_layer)
		else:
			self.conv_block1 = conv(c_num, c_num*2, 2, bias=use_bias, norm_layer=norm_layer)
			self.conv_block2 = conv(c_num*2, c_num*2, 1, bias=use_bias, norm_layer=norm_layer)

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
		# localization network
		self.localization = nn.Sequential(
			conv(input_ch, 8, 1, kernel_size=7),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True),
			conv(8, 10, 1, kernel_size=5),
			nn.MaxPool2d(2, stride=2),
			nn.ReLU(True)
		)

		# Regressor for the 3*2 affine matrix
		self.fc_loc = None 

	def _fc_loc(self, input):
		return nn.Sequential(
			nn.Linear(input, 32),
			nn.ReLU(True),
			nn.Linear(32, 3*2)
			)

	def forward(self, x, flow):
		_flow = self.localization(flow) 
		_flow_shape = _flow.shape
		_flow = _flow.view(-1, 10 * _flow_shape[-1] * _flow_shape[-2])

		if (self.fc_loc == None):
			self.fc_loc = self._fc_loc(10 * _flow_shape[-1] * _flow_shape[-2]) 
			#Initialize the weights/bias with identity transformation
			self.fc_loc[2].weight.data.zero_()
			self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

		theta = self.fc_loc(_flow)
		theta = theta.view(-1, 2, 3) # matrix for transformation

		grid = F.affine_grid(theta, x.size())
		x = F.grid_sample(x, grid)
		return x

"""
class for Feature Pyramid Networks
"""
class FPN(nn.Module):
	def __init__(self, N, ch_list):
		super(FPN, self).__init__()
		self.N = N
		self.ch = ch_list
		print("self.ch: {}".format(self.ch))

		# encoding layer - left to right
		self.conv = []
		for i in range(self.N):
			self.conv.append(BasicBlock(self.ch[i], i)) 
		
		self.toplayer = deconv(self.ch[-1]*2, 256, kernel_size=2, padding=0)

		# decoding layer - left to right
		self.deconv = []
		for i in range(self.N-1):
			self.deconv.append(deconv(self.ch[-1-i], 256, kernel_size=4, padding=1))

	def _upsample_add(self, x, y):
		_, _, H, W = y.size()
		return F.upsample(x, size=(H, W), mode="bilinear") + y

	def forward(self, input):
		encoder = []
		decoder = []

		encoder.append(input)
		for i in range(self.N):
			encoder.append(self.conv[i](encoder[-1]))

		decoder.append(self.toplayer(encoder[-1]))

		for i in range(self.N-1):
			x = self._upsample_add(decoder[-1], self.deconv[i](encoder[self.N - 1 - i]))
			decoder.append(x)
		return decoder

class FlowNet(nn.Module):
	def __init__(self, N, src_ch = 6, tar_ch = 3):
		super(FlowNet, self).__init__()
		self.N = N

		# define channel list
		self.src = []
		self.tar = []
		
		for i in range(self.N):
			if (i==0): 
				self.src.append(src_ch)
			else:
				self.src.append(2**(i+5)) # start with 32

		for i in range(self.N):
			if (i==0): 
				self.tar.append(tar_ch)
			else:
				self.tar.append(2**(i+5)) # start with 32

		self.SourceFPN = FPN(self.N, self.src)
		self.TargetFPN = FPN(self.N, self.tar)

		# list for Warp - left to right
		self.stn = []
		for i in range(self.N):
			self.stn.append(STN(2))

		# E layer - left to right
		self.E = []
		for i in range(self.N):
			self.E.append(predict_flow(MAX_CH * 2))

		self.lambda_struct = 10
		self.lambda_smt = 2
		self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

	def forward(self, src, tar):
		#input source,target image

		#  TODO: src, tar parse
		src_conv = self.SourceFPN(src) #[W, H, C]
		tar_conv = self.TargetFPN(tar) #[W, H, C]

		#concat for E4 to E1

		self.F = []
		self.F.append(self.E[0](torch.cat([src_conv[0], tar_conv[0]], 1))) #[W, H, 2]
		for i in range(self.N - 1):
			upsample_F = self.upsample(self.F[i]) #[2W, 2H, 2]
			warp = self.stn[i](src_conv[i+1], upsample_F)  
			concat = torch.cat([warp, tar_conv[i+1]], 1)
			self.F.append(upsample_F.add(self.E[i+1](concat)))

		last_F = self.upsample(self.F[-1])
		if DEBUG:
			print("*******************shape of src: {}, shape of last_F: {}*****************".format(src.shape, self.F[-1].shape))
		self.result = self.stn[-1](src, self.F[-1])
		cloth = self.result[:, :3, :, :]
		mask = self.result[:, 3:4, :, :]
		if DEBUG:
			print("**********shape of cloth: {}, shape of mask: {}***********".format(cloth.shape, mask.shape))
		# TODO: result parse
		return cloth, mask

	def backward(self):
		self.loss_roi_perc = loss_roi_perc(self.warp_seg, self.warp_cloth, self.t_seg, self.t_cloth)
		self.loss_struct = loss_struct(self.warp_seg, self.t_seg) 
		self.loss_smt = 0
		for i in self.N:
			self.loss_smt += loss_smt(self.F[i])

		self.loss_total =  self.loss_roi_perc + self.lambda_struct * self.loss_struct + self.lambda_smt * self.loss_smt

		loss_total.backward()

	def current_results():
		return { 'loss': self.loss_total }

	def loss_struct(self, src, tar):
		return nn.L1loss(src, tar)

	def loss_roi_perc(self, src_seg, src_cloth, tar_seg, tar_cloth):
		return VGGLoss(src_seg * src_cloth, tar_seg * tar_cloth)

	def loss_smt(self, mat):
		return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
				torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))
def test_FPN():
	return FPN(4, [3, 32, 64, 128])

def test_STN():
	return STN(256)

def test_FlowNet():
	return FlowNet(5, 4, 1)

def test():
	net = test_FlowNet()
	fms = net(Variable(torch.randn(1, 4, 192, 256)), Variable(torch.randn(1, 1, 192, 256)))

if __name__ == "__main__":
	test()
