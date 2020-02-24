import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pypreprocessor import pypreprocessor
pypreprocessor.parse()
 
PWD = '/home/fashionteam/ClothFlow/'
IS_TOPS = True
#define TEST

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    real_path = os.path.join(PWD, checkpoint_path)
    if not os.path.exists(real_path):
        print('[-] Checkpoint Load Error!')  
        exit() 
    model.load_state_dict(torch.load(real_path))	
    model.cuda()

def get_A(bs, H, W):
    A = np.array([[[1,0,0],[0,1,0]]]).astype(np.float32)
    A = np.concatenate([A]*bs,0)
    A = torch.from_numpy(A)
    net = nn.functional.affine_grid(A,(bs,2,W,H)).cuda()
    return net

def projection_grid(param, shape):
    alpha = torch.unsqueeze(torch.unsqueeze(param[:,0:1],2),3)
    beta = torch.unsqueeze(torch.unsqueeze(param[:,1:2],2),3)
    scale = torch.add(torch.unsqueeze(torch.unsqueeze(param[:,2:3],2),3), 1)
    theta = torch.unsqueeze(torch.unsqueeze(param[:,3:4],2),3)
    dis = 3
    # dis = torch.unsqueeze(torch.unsqueeze(param[:,4:],2),3)
    base = get_A(shape[0], shape[2], shape[3])
    base_X = base[:,:,:,0:1]
    base_Y = base[:,:,:,1:]
    # denominator = base_X * theta - dis * torch.sqrt(1 - theta ** 2)
    denominator = base_X * torch.sin(theta) - dis * torch.cos(theta)
    X = - 1 / scale * (dis * base_X / denominator + alpha)
    # Y = (dis * torch.sqrt(1 - theta ** 2)) / (-denominator * scale) * base_Y - beta / scale
    Y = (dis * torch.cos(theta)) / (-denominator * scale) * base_Y - beta / scale
    grid = torch.cat([X,Y], axis=3)
    return grid 

def WriteImage(writer,name,data,cnt,dataformats=None):
    if not (cnt % 50 == 0):
        return None
    
    data_ = (data.clone() + 1)*0.5
    #data_ = data_.cpu().clamp(0,255).detach().numpy().astype('uint8')
    #data_ = data_.swapaxes(1,2).swapaxes(2,3)
    #print(data_.shape)
    if dataformats is None:
        writer.add_images(name,data_,cnt)
    else:
        writer.add_images(name,data_,cnt,dataformats=dataformats)
