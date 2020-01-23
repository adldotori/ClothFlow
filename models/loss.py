import torch
import torch.nn as nn
from torchvision import models


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.loss = 0
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.criterion = nn.L1Loss()
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
        self.loss += self.lamdas[0]*self.criterion(h_relu1_X,h_relu1_Y.detach())

        h_relu2_X = self.slice2(h_relu1_X)
        h_relu2_Y = self.slice2(h_relu1_Y)
        self.loss += self.lamdas[1]*self.criterion(h_relu2_X,h_relu2_Y.detach())
        
        h_relu3_X = self.slice3(h_relu2_X)
        h_relu3_Y = self.slice3(h_relu2_Y)
        self.loss += self.lamdas[2]*self.criterion(h_relu3_X,h_relu3_Y.detach())

        h_relu4_X = self.slice4(h_relu3_X)
        h_relu4_Y = self.slice4(h_relu3_Y)
        self.loss += self.lamdas[3]*self.criterion(h_relu4_X,h_relu4_Y.detach())

        h_relu5_X = self.slice5(h_relu4_X)
        h_relu5_Y = self.slice5(h_relu4_Y)
        self.loss += self.lamdas[4]*self.criterion(h_relu5_X,h_relu5_Y.detach())

        return 0

    def zero_loss(self):
        self.loss = 0
    
    def get_loss(self):
        return self.loss


class VGGLoss(nn.Module):
    def __init__(self, layids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.lamdas = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        self.vgg.zero_loss()
        self.vgg(x,y)
        loss = self.vgg.get_loss()
        self.vgg.zero_loss()  
        return loss


class FlowLoss(nn.Module):
    def __init__(self, opt):
        super(FlowLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss()
        
	### original code ###
	#self.lambda_struct = 10
        #self.lambda_smt = 2

        self.lambda_struct = opt.struct_loss
        self.lambda_smt = opt.smt_loss
        self.lambda_roi = opt.perc_loss
	
    def forward(self, N, F, warp_mask, warp_cloth, tar_mask, tar_cloth):
        _loss_roi_perc = self.loss_roi_perc(
            warp_mask, warp_cloth, tar_mask, tar_cloth)
        _loss_struct = self.loss_struct(warp_mask, tar_mask)
        _loss_smt = self.loss_smt(F[0])
        for i in range(N-1):
            _loss_smt += self.loss_smt(F[i+1])

        return self.lambda_roi * _loss_roi_perc + self.lambda_struct * _loss_struct + self.lambda_smt * _loss_smt, _loss_roi_perc * self.lambda_roi, _loss_struct * self.lambda_struct, _loss_smt * self.lambda_smt

    def loss_struct(self, src, tar):
        return self.l1_loss(src, tar)

    def loss_roi_perc(self, src_mask, src_cloth, tar_mask, tar_cloth):
        ex_src_mask = src_mask.repeat(1, 3, 1, 1)
        ex_tar_mask = tar_mask.repeat(1, 3, 1, 1)
        return self.vgg_loss(ex_src_mask * src_cloth, ex_tar_mask * tar_cloth)

    def loss_smt(self, mat):
        return (torch.sum(torch.abs(mat[:, :, :, :-2] + mat[:, :, :, 2:] - 2*mat[:, :, :, 1:-1])) + \
            torch.sum(torch.abs(mat[:, :, :-2, :] + mat[:, :, 2:, :] - 2*mat[:, :, 1:-1, :])))/(mat.shape[2]*mat.shape[3])

        
