from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import numbers
import kornia
from kornia.filters.kernels import get_spatial_gradient_kernel2d, normalize_kernel2d
from torch.autograd import Variable

"""
SpatialGradient:
	Computes the first order image derivates in both x and y using Sobel operator
"""
class SpatialGradient(nn.Module):
	def __init__(self, order=1, normalized=True):
		super(SpatialGradient, self).__init__()
		self.normalized = normalized
		self.order = order
		self.mode = "sobel"
		self.kernel = get_spatial_gradient_kernel2d(self.mode, order)
		if self.normalized:
			self.kernel = normalize_kernel2d(self.kernel)
		return
	def forward(self, input):
		assert(len(input.shape) == 4)
		b, c, h, w = input.shape
		tmp_kernel = self.kernel.to(input.device).to(input.dtype).detach()
		kernel = tmp_kernel.unsqueeze(1).unsqueeze(1)

		kernel_flip = kernel.flip(-3)
		spatial_pad = [self.kernel.size(1) // 2,
							self.kernel.size(1) // 2,
							self.kernel.size(2) // 2,
							self.kernel.size(2) // 2]
		out_channels = 3 if self.order == 2 else 2
		padded_inp = F.pad(input.reshape(b*c, 1, h, w), spatial_pad, "replicate")[:, :, None]
		return F.conv3d(padded_inp, kernel_flip, padding=0).view(b, c, out_channels, h, w)

"""
Sobel Filter
"""
class Sobel(nn.Module):
	def __init__(self, normalized=True):
		super(Sobel, self).__init__()
		self.normalized = normalized
	def spatial_gradient(self, input, order=1, normalized=True):
		return SpatialGradient(order, normalized)(input)
	def one_hot(self, labels, class_count=3):
		y = torch.eye(class_count)
		return y[labels]
	def clipped_div(self, x, y, clip_val):
		eps = torch.ones(y.shape) * 1e-9
		y_tr = torch.where(y==0.0, eps, y)
		div = x / y_tr # check
		div_clipped = torch.clamp(div, -clip_val, clip_val)
		return div_clipped
	def forward(self, input):
		assert(len(input.shape) == 4)
		if (input.shape[1] == 3):

			R = input[:, :1, :, :]
			G = input[:, 1:2, :, :]
			B = input[:, 2:3, :, :]

			R_edges = self.spatial_gradient(R, normalized=self.normalized)
			G_edges = self.spatial_gradient(G, normalized=self.normalized)
			B_edges = self.spatial_gradient(B, normalized=self.normalized)

			R_gx = R_edges[:, :, 0]; R_gy = R_edges[:, :, 1]
			G_gx = G_edges[:, :, 0]; G_gy = G_edges[:, :, 1]
			B_gx = B_edges[:, :, 0]; B_gy = B_edges[:, :, 1]

			R_mag = torch.sqrt(R_gx*R_gx + R_gy*R_gy + 1e-9)
			G_mag = torch.sqrt(G_gx*G_gx + G_gy*G_gy + 1e-9)
			B_mag = torch.sqrt(B_gx*B_gx + B_gy*B_gy + 1e-9)

			#argmax
			_gx = torch.cat([R_gx, G_gx, B_gx], dim=1).transpose(1, 2).transpose(2, 3)
			_gy = torch.cat([R_gy, G_gy, B_gy], dim=1).transpose(1, 2).transpose(2, 3)
			_mag = torch.cat([R_mag, G_mag, B_mag], dim=1) # B, 3, H, W -> B, H, W, 3
			_mag = _mag.transpose(1, 2).transpose(2, 3)

			values, indices = _mag.max(3) # indices -> B, H, W
			hot = self.one_hot(indices, 3)
			gx = torch.sum(torch.mul(hot, _gx), 3)
			gy = torch.sum(torch.mul(hot, _gy), 3) # B, H, W
			
		else:
			# compute the x/y gradients
			edges = self.spatial_gradient(input, normalized=self.normalized)
			# unpack the edges
			gx = edges[:, :, 0]
			gy = edges[:, :, 1]
		
		magnitude = torch.sqrt(gx*gx + gy*gy + 1e-9)
		angle = self.clipped_div(gx, gy, 10.0)
		# div = torch.unsqueeze(3)
		# angle = torch.atan(torch.abs(gy) / torch.abs(gx))
		return magnitude, angle

"""
Apply gaussian smoothing on a tensor
filtering is performed seperately for each channel 
"""
class GaussianSmoothing(nn.Module):
	def __init__(self, channels=3, kernel_size=5, sigma=1, dim=2):
		super(GaussianSmoothing, self).__init__()
		if isinstance(kernel_size, numbers.Number):
			kernel_size = [kernel_size] * dim
		if isinstance(sigma, numbers.Number):
			sigma = [sigma] * dim
		kernel = 1
		meshgrids = torch.meshgrid([
					torch.arange(size, dtype=torch.float32)
					for size in kernel_size])
		for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
			mean = (size - 1) / 2

			kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
						 torch.exp(-((mgrid - mean) / std) ** 2 / 2 )
		kernel = kernel / torch.sum(kernel) # make sum to 1

		kernel = kernel.view(1, 1, *kernel.size())
		kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

		self.register_buffer('weight', kernel)
		self.group = channels

		self.conv = F.conv2d

	def forward(self, input):
		return self.conv(input, weight=self.weight, groups = self.group, padding=2)

def dir_filters():
	ver = np.asarray([[[0, 0, 0], [1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, 0]]], dtype=np.float32)
	hor = np.asarray([[[0, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0 , 1, 0]]], dtype=np.float32)
	dg1 = np.asarray([[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 1]]], dtype=np.float32)
	dg2 = np.asarray([[[0, 0, 1], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [1, 0, 0]]], dtype=np.float32)

	t1 = np.tan(np.pi/8)
	t3 = np.tan(3*np.pi/8)

	dir_dict = [
		dict(name="0",		min_angle=-t1,	max_angle=t1,	filter=hor),
		dict(name="pi4",	min_angle=t1,	max_angle=t3,	filter=dg1),
		dict(name="pi2",	min_angle=t3,	max_angle=-t3,	filter=ver),
		dict(name="3pi4", min_angle=-t3, max_angle=-t1, filter=dg2)
	]
	return dir_dict

class ClipValues(nn.Module):
	def __init__(self):
		super(ClipValues, self).__init__()
	def forward(self, x, min_value, max_value):
		if min_value < max_value:
			ret = (min_value <= x) & (x < max_value)
		else:
			ret = (min_value <= x) | (x < max_value)
		return ret.type(x.dtype)

class NonMaximumSuppression(nn.Module):
	def __init__(self): # mask = "hor" / "ver" -- np
		super(NonMaximumSuppression, self).__init__()
		self.maxpool = nn.MaxPool2d(kernel_size=3)

	def forward(self, x, mask):
		mask1 = torch.from_numpy(mask[0]).unsqueeze(0)
		mask2 = torch.from_numpy(mask[1]).unsqueeze(0)
		mask3 = torch.from_numpy(mask[2]).unsqueeze(0)

		batch, H, W = x.shape
		x = x.view(batch,1,H,W)
		output1 = F.conv2d(x, mask1.view(1,1,3,3),padding=1)
		output2 = F.conv2d(x, mask2.view(1,1,3,3),padding=1)
		output3 = F.conv2d(x, mask3.view(1,1,3,3),padding=1)
		temp = torch.cat([output1,output2,output3],1)
		out = torch.max(temp,axis=1)[0]
		return (out == x).type(torch.float32)*x

class Multiply(nn.Module):
	def __init__(self):
		super(Multiply, self).__init__()
	def forward(self, x):
		result = torch.ones(x[0].size())
		for t in x:
			result *= t
		return result

class Addition(nn.Module):
	def __init__(self):
		super(Addition, self).__init__()
	def forward(self, x):
		result = torch.zeros(x[0].size())
		for t in x:
			result += t
		return result

class ThresholdLayer(nn.Module):
	def __init__(self):
		super(ThresholdLayer, self).__init__()
	def relu_t(self, x, t):
		mint = torch.ones(x.shape) * 1e-9
		return torch.where(x>=t, x, mint)
	def forward(self, x, thres):
		outputs = self.relu_t(x, thres)
		return torch.sigmoid(outputs)

class Canny(nn.Module):
	def __init__(self):
		super(Canny, self).__init__()
		self.gauss = GaussianSmoothing()
		self.sobel = Sobel()
		self.addition = Addition()
		self.multiply = Multiply()
		self.clipvalues = ClipValues()
		self.suppression = NonMaximumSuppression()
		self.threshold = ThresholdLayer()
		self.suppressed = None
	def forward(self, x, low_threshold = 1, gauss=False):
		if (gauss == True): x = self.gauss(x)
		magnitude, direction = self.sobel(x)

		suppressed_lst = []
		for d in dir_filters():
			curr_dir = self.clipvalues(direction, d['min_angle'], d['max_angle'])
			curr_nms = self.suppression(magnitude, d['filter'])
			curr_suppressed = self.multiply([curr_nms, curr_dir])
			suppressed_lst.append(curr_suppressed)

		#Combine all directions
		suppressed = self.addition(suppressed_lst)
		self.suppressed = suppressed

		low_thresh = self.threshold(suppressed, low_threshold)
		
		return low_thresh



def clip_normalize(x,mini,maxi):
    x = (x - mini)/(maxi-mini)
    return x.clamp(0,1)

def f(x,deg=20):
    return x**deg

def h(x,deg=20):
    return (1-x)**deg

def g(x,deg=20):
    return 1 - f(x,deg) - h(x,deg)

def sample_gumbel(shape, eps=1e-30):
    U = torch.rand(shape)#.cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits #+ sample_gumbel(logits.size())
    #print(y.type())
    return y
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

def threshold_process(G,b_lower,b_upper,repeat = 10):
    # G.shape -> batch, channel, H, W
    G_ = clip_normalize(G,b_lower,b_upper)
    G_ = G_.transpose(1,2).transpose(2,3)
    f_ = f(G_)
    g_ = g(G_)
    h_ = h(G_)
    G_ = torch.cat([f_,g_,h_],3)
    a,b,c,d = G_.shape
    G_ = G_.view(a*b*c,d)
    one_hot = gumbel_softmax(G_,1e-30)
    one_hot = one_hot.view(a,b,c,d)
    one_hot = one_hot.transpose(2,3).transpose(1,2)
    W = torch.tensor([[[[2.0]],[[1.0]],[[0.0]]]])#.cuda()
    B = F.conv2d(one_hot,W)
    #return F.relu(B - 1)
    for i in range(repeat):
        B_ = F.max_pool2d(B,3,1,1) * B
        B_ = B_.clamp(0,2)
        if torch.sum(B_ - B) == 0:
            B = B_
            break
        else:
            B = B_
    B = B.clamp(0,2)
    return F.relu(B - 1)

class Canny_full(nn.Module):
    def __init__(self):
        super(Canny_full, self).__init__()
        self.canny = Canny()
    def forward(self, x, low_threshold = 50, high_threshold=100,gauss=False,repeat=8):
        lower = self.canny(x,low_threshold,gauss)
        suppressed = self.canny.suppressed
        return threshold_process(suppressed,low_threshold,high_threshold,repeat)

if __name__ == "__main__":
	torch.manual_seed(4)
	x = torch.rand(2, 3, 192, 256)
	model = Canny()
	model(x, 20)

