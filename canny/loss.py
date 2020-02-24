import torch
import torch.nn as nn
from torchvision import models

class WeightedMSE(nn.Module):
	def __init__(self, b_mult = 1.0):
		super(WeightedMSE, self).__init__()
		self.b_mult = b_mult

	def forward(self, y_true, y_pred):
		epsilon = 1e-07

		temp_y = y_true.clone()
		_ones = torch.ones(temp_y.shape).cuda()
		_zeros = torch.zeros(temp_y.shape).cuda()
		temp_y = torch.where(temp_y > 0.5, _ones, temp_y)
		temp_y = torch.where(temp_y <= 0.5, _zeros, temp_y)
		_mean = torch.mean(temp_y, dim=(1, 2, 3))

#		b = (torch.mean(y_true, dim=(1, 2, 3)) * self.b_mult).detach()

		_epsilon = (torch.ones(y_pred.shape) * epsilon).cuda()
		epsilon_ = (torch.ones(y_pred.shape) * (1-epsilon)).cuda()
		torch.where(y_pred < epsilon, _epsilon, y_pred)
		torch.where(y_pred > (1-epsilon), epsilon_, y_pred)

		zeros = torch.zeros(y_true.shape).cuda()
		ones = torch.ones(y_true.shape).cuda()
		
		# background covering
		zero_idx = torch.where(y_true==0, ones, zeros)
		zero_idx.type_as(torch.cuda.FloatTensor())
		m_neg = torch.mean(((y_pred ** 2) * zero_idx), dim=(1, 2, 3))

		# edges covering
		non_zero_idx = torch.where(y_true!=0, ones, zeros)
		non_zero_idx.type_as(torch.cuda.FloatTensor())
		m_pos = torch.mean(((y_pred - y_true) ** 2) * non_zero_idx, dim=(1, 2, 3))

		m = (1.0 - _mean) * m_pos + _mean * m_neg
		return torch.mean(m, dim=-1)

def gram_matrix(data):
    a, b, c, d = data.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = data.view(a, b, c * d)  # resise F_XL into \hat F_XL
    features_t = features.transpose(1,2)

    G = torch.matmul(features, features_t)  # compute the gram product

    del features
    del features_t

    return G

class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, condition, target):
        G1 = gram_matrix(condition)
        G2 = gram_matrix(target)
        loss = self.criterion(G1,G2)# G2 need detach 
        return loss

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.loss = 0
        self.percept = 0
        self.style = 0
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.criterion = nn.L1Loss()
        self.styleLoss = StyleLoss()
        self.lamdas = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.gammas = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, Y):
        h_relu1_X = self.slice1(X)
        h_relu1_Y = self.slice1(Y)
        self.percept += self.lamdas[0]*self.criterion(h_relu1_X,h_relu1_Y.detach())
        self.style += self.gammas[0]*self.styleLoss(h_relu1_X,h_relu1_Y.detach())

        h_relu2_X = self.slice2(h_relu1_X)
        h_relu2_Y = self.slice2(h_relu1_Y)
        self.percept += self.lamdas[1]*self.criterion(h_relu2_X,h_relu2_Y.detach())
        self.style += self.gammas[1]*self.styleLoss(h_relu2_X,h_relu2_Y.detach())
        
        h_relu3_X = self.slice3(h_relu2_X)
        h_relu3_Y = self.slice3(h_relu2_Y)
        self.percept += self.lamdas[2]*self.criterion(h_relu3_X,h_relu3_Y.detach())
        self.style += self.gammas[2]*self.styleLoss(h_relu3_X,h_relu3_Y.detach())

        h_relu4_X = self.slice4(h_relu3_X)
        h_relu4_Y = self.slice4(h_relu3_Y)
        self.percept += self.lamdas[3]*self.criterion(h_relu4_X,h_relu4_Y.detach())
        self.style += self.gammas[3]*self.styleLoss(h_relu4_X,h_relu4_Y.detach())

        h_relu5_X = self.slice5(h_relu4_X)
        h_relu5_Y = self.slice5(h_relu4_Y)
        self.percept += self.lamdas[4]*self.criterion(h_relu5_X,h_relu5_Y.detach())
        self.style += self.gammas[4]*self.styleLoss(h_relu5_X,h_relu5_Y.detach())

        self.loss = self.percept + self.style
        
        return 0

    def zero_loss(self):
        self.loss = 0
        self.percept = 0
        self.style = 0
    
    def get_loss(self):
        return self.loss
    
    def get_percept(self):
        return self.percept

    def get_style(self):
        return self.style

class cannyLoss(nn.Module):
	def __init__(self):
		super(cannyLoss, self).__init__()
		self.vgg = Vgg19()
		self.vgg.cuda()
		self.criterion = nn.L1Loss()
		self.lamdas = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
		self.gammas = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
		self.mse = WeightedMSE()
	def forward(self, x, y):
		self.vgg.zero_loss()
		x3 = torch.cat([x, x, x], 1)
		y3 = torch.cat([y, y, y], 1)
		self.vgg(x3, y3)
		style = self.vgg.get_style()
		self.vgg.zero_loss()
		mse = self.mse(x, y)
		return style, mse

if __name__=="__main__":
	loss = WeightedMSE()
	y_pred = torch.FloatTensor([[[[1, 0, 2, 2],[1, 1, 1, 1]],[[2, 2, 2, 2],[1, 4, 2, 2]]], [[[1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]]]]).cuda()
	print(y_pred.shape)
	loss.forward(y_pred, y_pred).backward()
	print(y_pred.grad)
