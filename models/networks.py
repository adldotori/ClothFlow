
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable

from torchvision import models
from loss import *

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

def deconv(in_channels, out_channels, activation = "relu"):
	if activation == "leaky":
		return nn.Sequential(
				nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
				nn.LeakyReLU(0.1, inplace=True)
				)
	else:
		return nn.Sequential(
				nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
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
		self.fc_loc = NULL 

	def fc_loc(self, input):
		return nn.Sequential(
			nn.Linear(input, 32),
			nn.ReLU(True),
			nn.Linear(32, 3*2)
			)

	def forward(self, x):
		x = self.localization(x) 
		x_shape = x.shape
		x = x.view(-1, 10 * x_shape[-1] * x_shape[-2])

		if (self.fc_loc == NULL):
			self.fc_loc = fc_loc(10 * x_shape[-1] * x_shape[-2]) 
			#Initialize the weights/bias with identity transformation
			self.fc_loc[2].weight.data.zero_()
			self.fc_loc[2].bias.data.copy_(torch.tensor[1, 0, 0, 0, 1, 0], dtype=torch.float)

		theta = self.fc_loc(x)
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
		self.ret_list = []
		self.ch = ch_list

		# encoding layer - left to right
		self.conv = []
		for i in range(N):
			self.conv.append(BasicBlock(self.ch[i])) 

		# decoding layer - left to right
		self.deconv = []
		for i in range(N-1):
			self.deconv.append(deconv(self.ch[-1-i], self.ch[-2-i]))
	 
		# upsampling layer - right to left
		self.upsample = []
		for i in range(N-1):
			self.upsample.append(upconv(self.ch[N-1-i], self.ch[N-2-i]))

	def forward(self, input):
		ret = input
		for i in range(self.N):
			ret = self.conv[i](ret)
			self.ret_list.append(ret)	
		return ret

	"""
	TODO: need to change concat to add - ok
	"""
	def deconv_forward(self, input, i):
		ret_dec = self.deconv[i](input)
		upsample = self.upsample[i](self.ret_list[self.n - i])
		add_enc = _upsample.add(self.ret_list[n-i-1])
		return torch.cat([ret_dec, add_ret], 1)

"""
TODO: max channel should be 256 - ok
"""
class FlowNet(nn.Module):
	def __init__(self, N, input_ch, h, w):
		self.N = N

		# define channel list
		self.ch = []
		for i in range(self.N):
			self.ch.append(min(input_ch*(i+1), 256))

		self.SourceFPN = FPN(self.N, self.ch)
		self.TargetFPN = FPN(self.N, self.ch)

		# list for Warp - left to right
		self.stn = []
		for i in range(self.N):
			self.stn.append(STN(self.ch[-2-i]))

		# E layer - left to right
		self.E = []
		for i in range(self.N):
			# multiple by 2 due to concat (Sn, Tn)
			self.E.append(predict_flow(self.ch[-1-i] * 2))

		self.lambda_struct = 10
		self.lambda_smt = 2

	def set_input(self, inputs):
		self.c_cloth = inputs["c_cloth"].cuda()
		self.t_cloth = inputs["t_cloth"].cuda()
		self.c_seg = inputs["c_seg"].cuda()
		self.t_seg = inputs["t_seg"].cuda()

	def forward(self, src, tar):
		#input source,target image
		src_conv = self.SourceFPN(src) #[W, H, C]
		tar_conv = self.TargetFPN(tar) #[W, H, C]

		self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

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
			upsample_F = self.upsample(self.F[i])
			warp = self.stn[i](torch.cat([src_conv, upsample_F], 1)) #concat?
			concat = torch.cat([warp, tar_conv], 1)
			self.F.append(upsample_F.add(self.E[i+1](concat)))

		last_F = self.upsample(self.F[-1])
		self.result = self.stn[-1](src)

	def backward(self):
		self.loss_roi_perc = loss_roi_perc(self.c_seg, self.c_cloth, self.t_seg, self.t_cloth)
		self.loss_struct = loss_struct(self.s_seg, self.t_seg) 
		self.loss_smt = 0
		for i in self.N:
			self.loss_smt += loss_smt(self.F[i])

		self.loss_total =  self.loss_roi_perc+ self.lambda_struct * self.loss_struct+ 
			self.lambda_smt * self.loss_smt

		loss_total.backward()

    def loss_struct(self, src, tar):
        return nn.L1loss(src, tar)

    def loss_roi_perc(self, src_seg, src_cloth, tar_seg, tar_cloth):
		return VGGLoss(src_seg * src_cloth, tar_seg * tar_cloth)

    def loss_smt(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

"""
def test_FPN():
	return FPN(4, [3, 6, 12, 24])

def test():
	net = test_FPN()
	fms = net(Variable(torch.randn(1, 3, 1024, 1024)))
	print(fms.shape)
"""
