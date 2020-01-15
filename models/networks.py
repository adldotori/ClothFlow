
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
		self.conv_block2 = conv(c_num, c_num*2, 1, bias=use_bias, norm_layer=norm_layer)

		if activation = "leaky":
			self.activation = nn.LeakyReLU(0.1, inplace=True)
		else:
			self.activaton = nn.ReLU()

	def forward(self, x):
		residual = self.activation(self.conv_block1(x))
		x = self.activation(self.conv_block2(residual))
		x = self.conv_block2(x)
		return self.activation(x+residual)

class FlowNet(nn.Module):
	def __init__(self, input_ch):
		# encoding layer
		self.conv1 = BasicBlock(input_ch)
		self.conv2 = BasicBlock(input_ch*2)
		self.conv3 = BasicBlock(input_ch*3)
		self.conv4 = BasicBlock(input_ch*4)
		self.conv5 = BasicBlock(input_ch*5)

		#decoding layer
		self.deconv4 = deconv(input_ch*5, input_ch*4)
		self.deconv3 = deconv(input_ch*4, input_ch*3)
		self.deconv2 = deconv(input_ch*3, input_ch*2)
		self.deconv1 = deconv(input_ch*2, input_ch)

		# encoding layer for F
		self.f5 = deconv(input_ch*10, input_ch*8)
		self.f4 = deconv(input_ch*8, input_ch*6)
		self.f3 = deconv(input_ch*6, input_ch*4)
		self.f2 = deconv(input_ch*4, input_ch*2)

	def forward(self, src, tar):
		"""
		predict warping image by putting into predict_flow
		"""

		#input source image
		src_conv1 = self.conv1.forward(src)
		src_conv2 = self.conv2.forward(src_conv1)
		src_conv3 = self.conv3.forward(src_conv2)
		src_conv4 = self.conv4.forward(src_conv3)
		src_conv5 = self.conv5.forward(src_conv4)

		#input target image
		tar_conv1 = self.conv1.forward(tar)
		tar_conv2 = self.conv2.forward(tar_conv1)
		tar_conv3 = self.conv3.forward(tar_conv2)
		tar_conv4 = self.conv4.forward(tar_conv3)
		tar_conv5 = self.conv5.forward(tar_conv4)

		#concat for E4 to E1
		E4 = torch.cat((src_conv5, tar_conv5), 1)
		output_f4 = self.f4(E4)
		
		flow4 = self.predict_flow4()
