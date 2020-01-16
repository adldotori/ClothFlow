
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

class FTN(nn.Module):
	def __init__(self, N, input_ch):
        self.N = N
		# encoding layer
        self.conv = []
        for i in range(N):
            self.conv.insert(0, BasicBlock(input_ch * (i+1)))

		#decoding layer
        self.deconv = []
        for i in range(N-1):
            self.deconv.append(deconv(input_ch * (N - i), input_ch * (N - i - 1)))

    def forward(self, input):
        ret = input
        for i in self.N:
            ret = self.conv[i].forward(ret)
        return ret
    
    def deconv_forward(self, input, i):
        return self.deconv[i].forward(input)

class FlowNet(nn.Module):
	def __init__(self, N, input_ch):
        self.N = N

		self.SourceFTN = FTN(N, input_ch)
        self.TargetFTN = FTN(N, input_ch)

		# E layer
        self.E = []
        for i in range(N):
            self.E.append(deconv(input_cv * (N - i), input_cv * (N - i -1)))

        self.upsample = []
        for i in range(N):
            self.upsample.append(nn.Upsample(scale_factor=4, mode='nearest'))

	def forward(self, src, tar):
		"""
		predict warping image by putting into predict_flow
		"""


		#input source,target image
		src_conv = self.SourceFTN.forward(src)
        tar_conv = self.TargetFTN.forward(tar)

		#concat for E4 to E1
        F = self.E.forward(torch.cat((src_conv, tar_conv), 1))
        for i in range(self.N):
            src_conv = self.SourceFTN.deconv_forward(src_conv, i)
            tar_conv = self.TargetFTN.deconv_forward(tar_conv, i)
            warp = WarpBlock.forward(src_conv, self.upsample[i].forward(F))
            concat = torch.cat(warp, tar_conv), 1)
            F = self.upsample[i].forward(F) + self.E.forward(concat)

