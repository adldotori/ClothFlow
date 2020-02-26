import pickle
import os.path as osp
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def np_logicaland(nplist):
    if len(nplist) == 2:
        return np.logical_and(nplist[0],nplist[1])
    return np.logical_and(np_logicaland(nplist[:-1]),nplist[-1])


def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_pkl(path,data):
    with open(path, "wb") as f:
        pickle.dump(data,f)

def meshgrid(H,W):
    nx, ny = (W, H)
    x = np.linspace(0, W-1, nx)
    y = np.linspace(0, H-1, ny)
    xv, yv = np.meshgrid(x, y)
    return xv,yv

def seperate(x,y,p1,p2,mode):
    p1x = p1[1]
    p1y = p1[0]
    p2x = p2[1]
    p2y = p2[0]
    
    if mode == "L":
        if p1x == p2x:
            # meaningless code
            return y >= p2y
        g = float((p2y-p1y)/(p2x-p1x))
        det = g*(x-p1x) + p1y
        return det >= y
        
    elif mode == "R":
        if p1x == p2x:
            # meaningless code
            return y >= p2y
        g = float((p2y-p1y)/(p2x-p1x))
        det = g*(x-p1x) + p1y
        return det <= y
        
    
    elif mode == "U":
        if p1x == p2x:
            return x <= p2x
        g = float((p2y-p1y)/(p2x-p1x))
        det = g*(x-p1x) + p1y
        if g > 0:
            return y >= det
        else:
            return y <= det
        
    
    elif mode == "D":
        if p1x == p2x:
            return x >= p2x
        g = float((p2y-p1y)/(p2x-p1x))
        det = g*(x-p1x) + p1y
        if g > 0:
            return y <= det
        else:
            return y >= det

def draw_square(pose_dic,H,W):
    UL = 16
    UR = 17
    DL = 2
    DR = 5 
    ul = pose_dic[UL]
    ur = pose_dic[UR]
    dl = pose_dic[DL]
    dr = pose_dic[DR]
    x,y = meshgrid(H,W)
    return np_logicaland([seperate(y,x,ul,ur,mode="D") ,seperate(y,x,dl,dr,mode="U"),seperate(y,x,ul,dl,mode="R"),seperate(y,x,ur,dr,mode="L")])

def draw_pentagon(pose_dic,H,W):
    UL = 16
    UR = 17
    DL = 2
    DR = 5
    ul = pose_dic[UL]
    ur = pose_dic[UR]
    dl = pose_dic[DL]
    dr = pose_dic[DR]
    torso = (np.array(pose_dic[3]) + np.array(pose_dic[6])) / 2
    x,y = meshgrid(H,W)
    uu = seperate(y,x,ul,ur,mode="D")
    ll = seperate(y,x,ul,dl,mode="R")
    rr = seperate(y,x,ur,dr,mode="L")
    tl = seperate(y,x,dl,torso,mode="U")
    tr = seperate(y,x,dr,torso,mode="U")
    return np_logicaland([uu,ll,rr,tl,tr])

def fullmake(mask,pose_dic):
    H, W = mask.shape
    penta = draw_pentagon(pose_dic,H,W)
    result = np.clip(penta + mask,0,1).astype(np.float32)
    return result

def setup(to_path,datamode,base_dir="/home/fashionteam/viton_resize/",namelist="viton_neckmask_"):
    parseroot = osp.join(base_dir,datamode,"image-parse")
    poseroot = osp.join(base_dir,datamode,"pose_pkl")
    namelist_path = namelist + datamode + ".pkl"
    nlist = load_pkl(namelist_path)
    pose_footer = "_0_keypoints.pkl"
    parse_footer = "_0.png"
    for i in range(len(nlist)):
        pose_name = nlist[i] + pose_footer
        pose_dic = load_pkl(osp.join(poseroot,pose_name))
        parse_name = nlist[i] + parse_footer
        mask = (np.array(Image.open(osp.join(parseroot,parse_name))) > 0).astype(np.float32)
        result = (fullmake(mask,pose_dic) * 255).astype(np.uint8)
        Image.fromarray(result).save(osp.join(to_path,datamode,parse_name))

if __name__ == '__main__':
    setup("/home/fashionteam/bodyshape","train")
        


