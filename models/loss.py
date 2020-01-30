import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import transform
from torchvision import models
from torch.autograd import Variable

class FeatureExtractor(nn.Module):
	def __init__(self, cnn, feature_layer=11):
		super(FeatureExtractor, self).__init__()
		self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])
	def normalize(self, tensors, mean, std):
		for tensor in tensors:
			for t, m, s in zip(tensor, mean, std):
				t.sub_(m).div_(s)
		return tensors
	def forward(self, x):
		if x.size()[1] == 1:
			x = x.expand(-1, 3, -1, -1)
		x = (x+1)*0.5
		x.data = self.normalize(x.data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		return self.features(x)

class VGGLoss(nn.Module):
	def __init__(self):
		super(VGGLoss, self).__init__()
		self.vgg19 = models.vgg19(pretrained=True)
		self.feature_extractor = FeatureExtractor(self.vgg19, feature_layer=35)
		self.mse = nn.MSELoss()
	
	def resize2d(self, img, size):
		with torch.no_grad():
			return (F.adaptive_avg_pool2d(Variable(img), size)).data

	def forward(self, x, y):
#		print("X: {}".format(x))
#		print("Y: {}".format(y))
		x = (x+1)*0.5
		y = (y+1)*0.5
		x = self.resize2d(x, (256, 256))
		y = self.resize2d(y, (256, 256))

		x = self.feature_extractor(x)
		y = self.feature_extractor(y).data
		return self.mse(x, y)


class FlowLoss(nn.Module):
    def __init__(self, opt):
        super(FlowLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.vgg_loss = VGGLoss()
        
        self.lambda_struct = opt.struct_loss
        self.lambda_smt = opt.smt_loss
        self.lambda_roi = opt.perc_loss
        self.bs = opt.batch_size
		  
    def get_A(self, bs, H, W):
        x = torch.linspace(-1, 1, H)
        y = torch.linspace(-1, 1, W)
        xv, yv = torch.meshgrid(x, y)
        xv = xv.view(1,H,W)
        yv = yv.view(1,H,W)
        A = torch.cat((xv,yv),0)
        A = A.view(1,2,H,W)
        A = torch.cat([A]*bs)
        
        return A.cuda()
    """
    COMMENT: roi_perc and struct to be calculated in every layer
    """

    def forward(self, N, F, warp_mask, warp_cloth, tar_mask, tar_cloth, warp_list):
        _loss_roi_perc = self.loss_roi_perc(
            warp_mask, warp_cloth, tar_mask, tar_cloth)
        
        """
        _loss_roi_perc = 0.0
        for i in range(N):
            _loss_roi_perc += self.loss_roi_perc(warp_list[i][:, 3:4, :, :], warp_list[i][:, :3, :, :], tar_mask, tar_cloth)
        """		 

        """ 
        _loss_struct = 0.0
        for i in range(N):
            _loss_struct += self.loss_struct(warp_list[i][:, 3:4, :, :], tar_mask)
        """
        _loss_struct = self.loss_struct(warp_mask, tar_mask)

        _loss_smt = self.loss_smt(F[0]-self.get_A(F[0].shape[0], F[0].shape[2], F[0].shape[3]))
        for i in range(N-1):
            _loss_smt += self.loss_smt(F[i+1]-self.get_A(F[i+1].shape[0], F[i+1].shape[2], F[i+1].shape[3]))

        return self.lambda_roi * _loss_roi_perc + self.lambda_struct * _loss_struct + self.lambda_smt * _loss_smt, _loss_roi_perc * self.lambda_roi, _loss_struct * self.lambda_struct, _loss_smt * self.lambda_smt

    def loss_struct(self, src, tar):
        """
        _tar = tar.transpose(0, 1).transpose(1, 2).transpose(2, 3)
        _tar = transform.resize(_tar.cpu().reshape(tar.shape[-2], tar.shape[-1], -1), (src.shape[-2], src.shape[-1]))
        _tar = _tar.reshape(src.shape[-2], src.shape[-1], 1, -1)
        _tar = torch.tensor(_tar)
        tar = _tar.transpose(2, 3).transpose(1, 2).transpose(0, 1)
        tar = tar.transpose(2, 3).transpose(1, 2).cuda()
        """
        return self.l1_loss(src, tar)

    def loss_detail(self, src, tar):
        return self.l2_loss(src, tar)

    def loss_roi_perc(self, src_mask, src_cloth, tar_mask, tar_cloth):
#        tar_mask.resize(src_mask.shape)
#        tar_cloth.resize(src_mask.shape)
        ex_src_mask = src_mask.repeat(1, 3, 1, 1)
        ex_tar_mask = tar_mask.repeat(1, 3, 1, 1)
        return self.vgg_loss(ex_src_mask*src_cloth, ex_tar_mask*tar_cloth)
    
    def loss_smt(self, mat):
        return (torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
				  torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))) / (mat.shape[2] * mat.shape[3])

        
