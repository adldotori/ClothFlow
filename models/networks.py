from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.autograd import Variable
from torchvision import models

from models.loss import *


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

	def __init__(self, c_num, norm="batch", activation="relu"):
		super(BasicBlock, self).__init__()
		
		if norm == "batch":
			norm_layer = nn.BatchNorm2d
			use_bias = False
		elif norm == "instance":
			norm_layer = nn.InstanceNorm2d
			use_bias = True
		else:
			raise NotImplementedError()

		if (c_num == 3 or c_num == 1):
			self.conv_block1 = conv(c_num, 32, 2, bias=use_bias, norm_layer=norm_layer)
			self.conv_block2 = conv(32, 32, 1, bias=use_bias, norm_layer=norm_layer)
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

TODO: change channels to be dependent on input channels
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

	def forward(self, x):
		_x = self.localization(x) 
		_x_shape = _x.shape
		_x = _x.view(-1, 10 * _x_shape[-1] * _x_shape[-2])

		if (self.fc_loc == None):
			self.fc_loc = self._fc_loc(10 * _x_shape[-1] * _x_shape[-2]) 
			#Initialize the weights/bias with identity transformation
			self.fc_loc[2].weight.data.zero_()
			self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

		theta = self.fc_loc(_x)
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
		self.decoder = []
		self.ch = ch_list
		print("self.ch: {}".format(self.ch))

		# encoding layer - left to right
		self.conv = []
		for i in range(self.N):
			self.conv.append(BasicBlock(self.ch[i])) 
		
		self.toplayer = deconv(self.ch[-1], 256, kernel_size=1, padding=0)

		# decoding layer - left to right
		self.deconv = []
		for i in range(self.N-2):
			self.deconv.append(deconv(self.ch[-2-i], 256, kernel_size=3, padding=1))

		# upsampling layer - right to left
		self.upsample = []
		for i in range(self.N-1):
			self.upsample.append(upconv(self.ch[self.N-1-i], self.ch[self.N-2-i]))

	def _upsample_add(self, x, y):
		_, _, H, W = y.size()
		return F.upsample(x, size=(H, W), mode="bilinear") + y

	def forward(self, input):
		ret = input
		for i in range(self.N):
			ret = self.conv[i](ret)
			print("shape of ret {} is {}".format(i, ret.shape))
		decoder[0] = self.toplayer(ret)
		for i in range(self.N-1):
			decoder[i+1] = self._upsample_add(decoder[i], self.deconv[i])
		return self.decoder

	# def deconv_forward(self, input, i):
	# 	ret_dec = self.deconv[i](input)
	# 	upsample = self.upsample[i](self.ret_list[self.N - i])
	# 	add_enc = _upsample.add(self.ret_list[self.N-i-1])
	# 	return torch.cat([ret_dec, add_ret], 1)

"""
TODO: max channel should be 256 - ok
"""
class FlowNet(nn.Module):
	def __init__(self, N, input_ch = 3):
		super(FlowNet, self).__init__()
		self.N = N

		# define channel list
		self.ch = []
		
		for i in range(self.N):
			if (i==0): 
				self.ch.append(input_ch)
			else:
				self.ch.append(min(2**(i+4), 256)) # start with 32

		self.SourceFPN = FPN(self.N, self.ch)
		self.TargetFPN = FPN(self.N, self.ch)

		# list for Warp - left to right
		self.stn = []
		for i in range(self.N-1):
			self.stn.append(STN(self.ch[-2-i]))

		# E layer - left to right
		self.E = []
		for i in range(self.N):
			# TODO: i=0일 때 -로 들어감
			# multiple by 2 due to concat (Sn, Tn)
			self.E.append(predict_flow(self.ch[-1-i] * 2))

		self.lambda_struct = 10
		self.lambda_smt = 2
		self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

	def forward(self, src, tar):
		#input source,target image

		#  TODO: src, tar parse
		src_conv = self.SourceFPN(src) #[W, H, C]
		tar_conv = self.TargetFPN(tar) #[W, H, C]


		"""
		TODO: change code
		E => predict_flow - ok
		"""
		#concat for E4 to E1

		self.F = []
		self.F.append(self.E[0](torch.cat([src_conv, tar_conv], 1))) #[W, H, 2]
		for i in range(self.N - 1):
			src_conv = self.SourceFPN.deconv_forward(src_conv, i) #[2W, 2H, C]
			tar_conv = self.TargetFPN.deconv_forward(tar_conv, i) #[2W, 2H, C]
			upsample_F = self.upsample(self.F[i]) #[2W, 2H, 2]
			warp = self.stn[i](torch.cat([src_conv, upsample_F], 1)) #concat? 
			concat = torch.cat([warp, tar_conv], 1)
			self.F.append(upsample_F.add(self.E[i+1](concat)))

		last_F = self.upsample(self.F[-1])
		self.result = self.stn[-1](torch.cat([src, last_F], 1))
		# TODO: result parse
		return self.result

	def backward(self):
		self.loss_roi_perc = loss_roi_perc(self.warp_seg, self.warp_cloth, self.t_seg, self.t_cloth)
		self.loss_struct = loss_struct(self.warp_seg, self.t_seg) 
		self.loss_smt = 0
		for i in self.N:
			self.loss_smt += loss_smt(self.F[i])

		self.loss_total =  self.loss_roi_perc+ self.lambda_struct * self.loss_struct + self.lambda_smt * self.loss_smt

		loss_total.backward()

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
	return STN(3)

def test_FlowNet():
	return FlowNet(4, 3)

def test():
	net = test_FPN()
	# net = test_STN()
	# net = test_FlowNet()
	fms = net(Variable(torch.randn(1, 3, 1024, 1024)))
	# fms = net(Variable(torch.randn(1, 3, 1024, 1024), torch.randn(1, 3, 1024, 1024)))
	print(fms.shape)

if __name__ == "__main__":
	test()
