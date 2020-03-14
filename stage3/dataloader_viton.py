#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
from PIL import ImageDraw
from pathlib import Path
import os.path as osp
import numpy as np
import json
import pickle
import sys
sys.path.append("..")
import neckmake

#import time

INPUT_SIZE = (512, 512)

def load_pkl(name):
    with open(name,"rb") as f:
        data = pickle.load(f)
    return data

def naming(file_name):
    return file_name[:-6]

class CFDataset(data.Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, opt):
        super(CFDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.stage = opt.stage # GMM or TOM
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_1ch = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
        self.warped_cloth_path = opt.dataroot_cloth
        self.warped_mask_path = opt.dataroot_mask
        
        # load data list
        #self.image_files = os.listdir(osp.join(self.data_path,"image"))
        self.image_files = load_pkl(osp.join("/home/fashionteam/ClothFlow/","viton_real_"+self.datamode+".pkl"))

    def name(self):
        return "CFDataset"

    def __getitem__(self, index):
        name = self.image_files[index]

        # cloth image & cloth mask

        path_cloth = osp.join(self.data_path,"cloth",name+"_1.jpg")
        path_mask = osp.join(self.data_path, "cloth-mask",name+"_1.jpg")
        cloth = Image.open(path_cloth)
        cloth_mask = Image.open(path_mask)

        path_image = osp.join(self.data_path, "image",name+"_0.jpg") # condition image path

        image = Image.open(path_image)
        warped = Image.open(osp.join(self.warped_cloth_path,name+".jpg"))
        warped_mask = Image.open(osp.join(self.warped_mask_path,name+".jpg"))
        warped_mask = (np.array(warped_mask) > 0).astype(np.float32)
        #warped_mask = torch.from_numpy(warped_mask)
        #warped_mask = warped_mask.unsqueeze_(0)


        H, W, C = np.array(image).shape###

        c_mask_array = np.array(cloth_mask)
        c_mask_array = (c_mask_array > 0).astype(np.float32)
        c_mask = torch.from_numpy(c_mask_array)
        cloth_mask = c_mask.unsqueeze_(0)

        original_image = image #

        cloth_ = self.transform(cloth)
        image = self.transform(image)
        warped = self.transform(warped)

        # parsing and pose path
        path_seg = osp.join(self.data_path, "image-seg", name+"_0.png")
        path_pose = osp.join(self.data_path, "pose_pkl",name+"_0.pkl")

        # segment processing
        seg = Image.open(path_seg)
        parse = self.transform_1ch(seg)
        parse_array = np.array(seg)

        shape = (parse_array > 0).astype(np.float32)  # condition body shape
        head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 2).astype(np.float32) + \
                 (parse_array == 4).astype(np.float32) + \
                 (parse_array == 13).astype(np.float32)
        cloth = (parse_array == 5).astype(np.float32) + \
                  (parse_array == 6).astype(np.float32) + \
                  (parse_array == 7).astype(np.float32)

        arms = (parse_array == 15).astype(np.float32) + \
                  (parse_array == 14).astype(np.float32)
        
        pants = (parse_array == 9).astype(np.float32) + \
                  (parse_array == 12).astype(np.float32)

        cloth = torch.from_numpy(cloth)
        warped_mask = torch.from_numpy(warped_mask)
        arms = torch.from_numpy(arms)
        pants = torch.from_numpy(pants)
        head = torch.from_numpy(head)

        with open(path_pose, 'rb') as f:
            pose_label = pickle.load(f)
        pose_data = pose_label

        shape = neckmake.fullmake(shape,pose_data)
        shape = torch.from_numpy(shape)
        shape.unsqueeze_(0)

        # rgb_array = np.full((3,INPUT_SIZE[0],INPUT_SIZE[1]), np.random.rand(3))
        # rgb_array = rgb_array.astype(np.float32)
        # rand = torch.from_numpy(rgb_array)
        image = image * shape + (1 - shape) * 0
        
        crop_head = image * head + (1 - head)
        crop_cloth = image * cloth + (1 - cloth)
        off_cloth = image * (1-cloth - arms) + cloth + arms
        crop_arms = image * arms + (1-arms)
        crop_pants = (image * pants + (1-pants)) * (1 - warped_mask) + warped_mask
        cloth.unsqueeze_(0)
        head.unsqueeze_(0)

        point_num = 18
        pose_map = torch.zeros(point_num, H, W)
        r = self.radius
        pose = Image.new('L', (W, H))
        pose_draw = ImageDraw.Draw(pose)
        for i in range(point_num):
            one_map = Image.new('L', (W, H))
            draw = ImageDraw.Draw(one_map)
            if i in pose_data.keys():
                pointx = pose_data[i][0]
                pointy = pose_data[i][1]
            else:
                pointx = -1
                pointy = -1

            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            one_map = self.transform_1ch(one_map)
            pose_map[i] = one_map[0]

        result = {
            'cloth': cloth_,# original cloth
            'cloth_mask': cloth_mask,# original cloth mask
            'image': image,  # source image
            'head': crop_head,# cropped head from source image
            'pose': pose_map,#pose map
            'tar_mask': shape,
            'crop_cloth': crop_cloth,#cropped cloth
            'crop_cloth_mask' : cloth,#cropped cloth mask
            'name' : name,
            'warped' : warped,
            'off_cloth': off_cloth,#source image - cloth
            'pants_mask': pants,
            'crop_pants': crop_pants,
            'parse': parse,
            'head_mask': head,
            }
        return result

    def __len__(self):
        return len(self.image_files)

class CFDataLoader(object):
    def __init__(self, opt, dataset):
        super(CFDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "/home/fashionteam/Taeho/VITON/train")
    parser.add_argument("--datamode", default = "")
    parser.add_argument("--stage", default = "PGP")
    parser.add_argument("--data_list", default = "MVCup_pair.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    
    opt = parser.parse_args()
    dataset = CFDataset(opt)
    data_loader = CFDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed; embed()

