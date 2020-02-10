import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import numpy as np

def get_A(bs, H, W):
    A = np.array([[[1,0,0],[0,1,0]]]).astype(np.float32)
    A = np.concatenate([A]*bs,0)
    A = torch.from_numpy(A)
    net = nn.functional.affine_grid(A,(bs,2,H,W)).cuda()
    net = net.transpose(2,3).transpose(1,2)
    return net


def Zscore(flow,func,scm,wcm,eps=1e-6):
    """
    input : flow -> distribution to test, c -> condition distribution, scm -> src mask, wcm -> warped mask
    size : flow -> (B,2,192*256), func -> function of x and y, scm,wcm -> (B,1,256,192) (bool-like type)
    e.g. func = lambda x,y: x**2 + y**2 where x and y are tensor of which size is (B,256*192) and
    output size is (B,256*192)
    output-size : (B,)
    """
    B,C,H,W = scm.shape # C = 1
    scm = scm.view(B,H*W)
    n_src = torch.sum(scm,axis=1) #(B,)
    scm = scm.view(B,1,H*W)
    scm = torch.cat([scm]*2,1) #B,2,H*W
    A = get_A(B,H,W) # B,2,H,W
    A = A.view(B,2,H*W) #B,2,H*W
    CD = scm * A
    CD_x = CD[:,0,:]
    CD_y = CD[:,1,:]
    expect_of_CD = torch.sum(func(CD_x,CD_y),axis=1) / (n_src+eps)
    expect_of_CD2 = torch.sum(func(CD_x,CD_y)**2,axis=1) / (n_src+eps)

    wcm = torch.eq(torch.ones(B,H*W).cuda(),wcm.view(B,H*W))
    n_warp = torch.sum(wcm,axis=1)
    wcm = wcm.view(B,1,H*W)
    wcm = torch.cat([wcm]*2,1)
    FD = flow.view(B,2,H*W) * M
    FD_x = FD[:,0,:]
    FD_y = FD[:,1,:]
    expect_of_FD = torch.sum(func(FD_x,FD_y),axis=1) / (n_warp+eps)
    z_score = (expect_of_FD - expect_of_CD)/torch.sqrt((expect_of_CD2 - expect_of_CD)/(n_warp+eps))
    
    return torch.abs(z_score)

def absFlow(flow):
    B,C,H,W = flow.shape # C = 2
    A = get_A(B,H,W)
    F = flow - A
    F = F**2
    F = torch.sum(F,axis=1)
    F = torch.sqrt(F)
    return torch.mean(F)




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
        self.lambda_stat = opt.stat_loss
        self.lambda_abs = opt.abs_loss
        self.eps = 1e-5
		  
    def get_A(self,bs, H, W):
        A = np.array([[[1,0,0],[0,1,0]]]).astype(np.float32)
        A = np.concatenate([A]*bs,0)
        A = torch.from_numpy(A)
        net = nn.functional.affine_grid(A,(bs,2,H,W)).cuda()
        net = net.transpose(2,3).transpose(1,2)
        return net
	
    def forward(self, N, F, warp_mask, warp_cloth, tar_mask, tar_cloth,src_mask): 
        _loss_roi_perc = self.loss_roi_perc(
            warp_mask, warp_cloth, tar_mask, tar_cloth)
        _loss_struct = self.loss_struct(warp_mask, tar_mask)
        _loss_smt = self.loss_smt(F[0]-self.get_A(F[0].shape[0], F[0].shape[2], F[0].shape[3]))
        for i in range(N-1):
            _loss_smt += self.loss_smt(F[i+1]-self.get_A(F[i+1].shape[0], F[i+1].shape[2], F[i+1].shape[3]))
        
        if self.lambda_stat == -1:
            _loss_stat = torch.tensor(0)
        else:
            _loss_stat = self.stat_loss(F[0],src_mask,warp_mask)

        if self.lambda_abs == -1:
            _loss_abs = torch.tensor(0)
        else:
            _loss_abs = absFlow(F[0])

        return self.lambda_roi * _loss_roi_perc + self.lambda_struct * _loss_struct + self.lambda_smt * _loss_smt + self.lambda_stat * _loss_stat + self.lambda_abs * _loss_abs , _loss_roi_perc * self.lambda_roi, _loss_struct * self.lambda_struct, _loss_smt * self.lambda_smt, _loss_stat,_loss_abs
        
    def loss_struct(self, src, tar,version="MS"):
        if version == "MS":
            return torch.mean(torch.abs(F.leaky_relu(tar-src,0.1764)))
        if version =="Original":
            return self.l1_loss(src, tar)
        return self.l1_loss(src, tar)

    def loss_detail(self, src, tar):
        return self.l2_loss(src, tar)

    def loss_roi_perc(self, src_mask, src_cloth, tar_mask, tar_cloth):
#       print("src_cloth: {}, tar_cloth: {}".format(src_cloth, tar_cloth))
        ex_src_mask = src_mask.repeat(1, 3, 1, 1)
        ex_tar_mask = tar_mask.repeat(1, 3, 1, 1)
        return self.vgg_loss(ex_src_mask*src_cloth, ex_tar_mask*tar_cloth)
    
    def loss_smt(self, mat):
        return (torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
				  torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))) / (mat.shape[2] * mat.shape[3])

        # return (torch.sum(torch.abs(mat[:, :, :, :-2] + mat[:, :, :, 2:] - 2*mat[:, :, :, 1:-1])) + \
		# 		  torch.sum(torch.abs(mat[:, :, :-2, :] + mat[:, :, 2:, :] - 2 * mat[:, :, 1:-1, :]))) / (mat.shape[2] * mat.shape[3])
    
    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                                grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                                prob_interpolated.size()),
                                create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
        
    def ClothDistribution(self,scm):
        """
        input : source cloth mask -> B,1,H,W
        output : tensor of expectation of x,y,x2,y2,x4,y4,xy,x2y2,xy2,x2y,x2y4,x4y2
                    size : B*12
        """
        #result = []
        eps = self.eps
        B,C,H,W = scm.shape # C = 1
        scm = scm.view(B,H*W)
        n = torch.sum(scm,axis=1) #(B,)
        scm = scm.view(B,1,H*W)
        scm = torch.cat([scm]*2,1) #B,2,H*W
        A = self.get_A(B,H,W) # B,2,H,W
        A = A.view(B,2,H*W) #B,2,H*W
        #print(scm.shape,A.shape)
        CD = scm * A
        x_ = torch.sum(CD[:,0,:],axis=1) / n
        y_ = torch.sum(CD[:,1,:],axis=1) / n
        x_ = x_.view(-1,1)
        y_ = y_.view(-1,1)
        result = torch.cat([x_,y_],axis=1)
        x2_ = torch.sum(CD[:,0,:]**2,axis=1) / n
        y2_ = torch.sum(CD[:,1,:]**2,axis=1) / n
        x2_ = x2_.view(-1,1)
        y2_ = y2_.view(-1,1)
        result = torch.cat([result,x2_,y2_],axis=1)
        x4_ = torch.sum(CD[:,0,:]**4,axis=1) / n
        y4_ = torch.sum(CD[:,1,:]**4,axis=1) / n
        x4_ = x4_.view(-1,1)
        y4_ = y4_.view(-1,1)
        result = torch.cat([result,x4_,y4_],axis=1)
        xy_ = torch.sum(CD[:,0,:]*CD[:,1,:],axis=1) / n
        xy_ = xy_.view(-1,1)
        x2y2_ = torch.sum((CD[:,0,:]**2)*(CD[:,1,:]**2),axis=1) / n
        x2y2_ = x2y2_.view(-1,1)
        result = torch.cat([result,xy_,x2y2_],axis=1)
        xy2_ = torch.sum((CD[:,0,:])*(CD[:,1,:]**2),axis=1) / n
        x2y_ = torch.sum((CD[:,0,:]**2)*(CD[:,1,:]),axis=1) / n
        xy2_ = xy2_.view(-1,1)
        x2y_ = x2y_.view(-1,1)
        result = torch.cat([result,xy2_,x2y_],axis=1)
        x2y4_ = torch.sum((CD[:,0,:]**2)*(CD[:,1,:]**4),axis=1) / n
        x4y2_ = torch.sum((CD[:,0,:]**4)*(CD[:,1,:]**2),axis=1) / n
        x2y4_ = x2y4_.view(-1,1)
        x4y2_ = x4y2_.view(-1,1)
        result = torch.cat([result,x2y4_,x4y2_],axis=1)
        return result

    def stat_loss(self,Final_F,scm,warp_mask):
        eps = self.eps
        B,C,H,W = warp_mask.shape # C = 1
        M = torch.eq(torch.ones(B,H*W).cuda(),warp_mask.view(B,H*W))
        #M = M.view(B,H*W)
        n = torch.sum(M,axis=1) #(B,)
        M = M.view(B,1,H*W)
        M = torch.cat([M]*2,1) #B,2,H*W
        FD = Final_F.view(B,2,H*W) * M
        CD = self.ClothDistribution(scm)
        x_ = torch.sum(FD[:,0,:],axis=1) / n
        y_ = torch.sum(FD[:,1,:],axis=1) / n
        x_ = x_.view(-1,1)
        y_ = y_.view(-1,1)
        x2_ = torch.sum(FD[:,0,:]**2,axis=1) / n
        y2_ = torch.sum(FD[:,1,:]**2,axis=1) / n
        x2_ = x2_.view(-1,1)
        y2_ = y2_.view(-1,1)
        xy_ = torch.sum(FD[:,0,:]*FD[:,1,:],axis=1) / n
        xy_ = xy_.view(-1,1)
        xy2_ = torch.sum((FD[:,0,:])*(FD[:,1,:]**2),axis=1) / n
        x2y_ = torch.sum((FD[:,0,:]**2)*(FD[:,1,:]),axis=1) / n
        xy2_ = xy2_.view(-1,1)
        x2y_ = x2y_.view(-1,1)
        #print(n,torch.sqrt((CD[:,2]-CD[:,0]**2)/n),torch.sqrt((CD[:,3]-CD[:,1]**2)/n),torch.sqrt((CD[:,10]-CD[:,8]**2)/n))
        if torch.sum(((CD[:,2]-CD[:,0]**2) < 0) | ((CD[:,3]-CD[:,1]**2) < 0) | ((CD[:,4]-CD[:,2]**2) < 0)  | ((CD[:,5]-CD[:,3]**2) < 0) | ((CD[:,7]-CD[:,6]**2) < 0) | ((CD[:,10]-CD[:,8]**2) < 0) | ((CD[:,11]-CD[:,9]**2) < 0)) > 0:
            print("Negative error")
        Dx = torch.pow((x_ - CD[:,0])/(torch.sqrt((CD[:,2]-CD[:,0]**2)/(n+eps))+eps),2)
        Dy = torch.pow((y_ - CD[:,1])/(torch.sqrt((CD[:,3]-CD[:,1]**2)/(n+eps))+eps),2)
        Dx2 = torch.pow((x2_ - CD[:,2])/(torch.sqrt((CD[:,4]-CD[:,2]**2)/(n+0.3*eps))+0.3*eps),2)
        Dy2 = torch.pow((y2_ - CD[:,3])/(torch.sqrt((CD[:,5]-CD[:,3]**2)/(n+0.3*eps))+0.3*eps),2)
        Dxy = torch.pow((xy_ - CD[:,6])/(torch.sqrt((CD[:,7]-CD[:,6]**2)/(n+0.3*eps))+0.3*eps),2)
        Dxy2 = torch.pow((xy2_ - CD[:,8])/(torch.sqrt((CD[:,10]-CD[:,8]**2)/(n+0.1*eps))+0.1*eps),2)
        Dx2y = torch.pow((x2y_ - CD[:,9])/(torch.sqrt((CD[:,11]-CD[:,9]**2)/(n+0.1*eps))+0.1*eps),2)
        return torch.mean(Dx + Dy + Dx2 + Dy2 + Dxy + Dxy2 + Dx2y)

