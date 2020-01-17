
"""
Two feature pyramid networks - source FPN, target FPN
N encoding layers => downsample conv with stride 2 followed by one residual block
N = 4 or 5
"""

def conv(in_channels, out_channels, stride, kernel_size=3, padding=0, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
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


def BasicBlock(nn.Module):
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

		if activation = "leaky":
			self.activation = nn.LeakyReLU(0.1, inplace=True)
		else:
			self.activaton = nn.ReLU()

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
	# NEED to check channel number ==> output should be 3*@
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
		x = self.localization(x.view(28, 28))) #resize x to (28, 28)
		x_shape = x.shape
		x = x.view(-1, 10 * x_shape[-1] * x_shape[-2])

		self.fc_loc = fc_loc(10 * x_shape[-1] * x_shape[-2]) # 가능..?
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
class FTN(nn.Module):
	def __init__(self, N, ch_list):
		super(FTN, self).__init__()
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
			self.upsample.append(upsample(self.ch[N-i], self.ch[N-1-i]))

	def forward(self, input):
		ret = input
		for i in self.N:
			ret = self.conv[i](ret)
			self.ret_list.append(ret)	
		return ret

	"""
	TODO: need to change concat to add - ok
	"""
	 def deconv_forward(self, input, i):
		ret_dec = self.deconv[i](input)
		upsample = self.upsample[i](self.ret_list[self.N - i])
		add_enc = _upsample.add(self.ret_list[N-i-1])
		return torch.cat([ret_dec, add_ret], 1)
	
"""
TODO: max channel should be 256 - ok
"""
class FlowNet(nn.Module):
	def __init__(self, N, input_ch, h, w):
		self.N = N

		# define channel list
		self.ch = []
		for i in range(N):
			self.ch.append(min(input_ch*(i+1), 256))

		self.SourceFTN = FTN(N, self.ch)
		self.TargetFTN = FTN(N, self.ch)

		# list for Warp - left to right
		self.stn = []
		for i in range(N-1):
			self.stn.append(STN(self.ch[-2-i]))

		# E layer - left to right
		self.E = []
		for i in range(N):
			# multiple by 2 due to concat (Sn, Tn)
			self.E.append(predict_flow(self.ch[-1-i] * 2))

		self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

	def forward(self, src, tar):
		#input source,target image
		src_conv = self.SourceFTN(src) #[W, H, C]
      tar_conv = self.TargetFTN(tar) #[W, H, C]

		"""
		TODO: change code
				E => predict_flow 사용 - ok
		"""
		#concat for E4 to E1
		F = self.E[0](torch.cat([src_conv, tar_conv], 1)) #[W, H, 2]
      for i in range(self.N - 1):
			src_conv = self.SourceFTN.deconv_forward(src_conv, i) #[2W, 2H, C]
			tar_conv = self.TargetFTN.deconv_forward(tar_conv, i) #[2W, 2H, C]
			upsample_F = self.upsample(F)
			warp = self.stn[i](torch.cat([src_conv, upsample_F], 1)) #concat?
			concat = torch.cat([warp, tar_conv], 1)
			F = upsample_F.add(self.E[i+1](concat))
		return F
		

