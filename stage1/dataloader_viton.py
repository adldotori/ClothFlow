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
import random

import neckmake
#import time
INPUT_SIZE = (512, 512)

def load_pkl(name):
    with open(name,"rb") as f:
        data = pickle.load(f)
    return data
def naming(file_name):
    return file_name[:6]

class CFDataset(data.Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, opt, is_tops=True):
        super(CFDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_1ch = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
        
        # load data list
        self.image_files = load_pkl(osp.join("/home/fashionteam/ClothFlow/stage1","stage1_dat.pkl"))
        # self.image_files = os.listdir('/home/fashionteam/')

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
        H, W, C = np.array(image).shape###

        c_mask_array = np.array(cloth_mask)
        c_mask_array = (c_mask_array > 25).astype(np.float32)
        c_mask = torch.from_numpy(c_mask_array)
        cloth_mask = c_mask.unsqueeze_(0)

        original_image = image #

        cloth_ = self.transform(cloth)
        image = self.transform(image)

        aug_image = torch.zeros(image.shape)
        w = random.randint(-150,150)
        h = random.randint(0,200)
        if w < 0:
            aug_image[:, h:, :INPUT_SIZE[0]+w] = image[:, :INPUT_SIZE[1]-h, -w:]
        else:
            aug_image[:, h:, w:] = image[:, :INPUT_SIZE[1]-h, :INPUT_SIZE[0]-w]
        image = aug_image

        # parsing and pose path
        path_seg = osp.join(self.data_path, "image-seg", name+"_0.png")
        path_mask = osp.join(self.data_path, "image-mask", name+"_0.png")
        path_pose = osp.join(self.data_path, "pose_pkl",name+"_0.pkl")

        # segment processing
        seg = Image.open(path_seg)
        parse = self.transform_1ch(seg)
        parse_array = np.array(seg)
        aug_parse_array = np.zeros(parse_array.shape)
        if w < 0:
            aug_parse_array[h:, :INPUT_SIZE[0]+w] = parse_array[:INPUT_SIZE[1]-h, -w:]
        else:
            aug_parse_array[h:, w:] = parse_array[:INPUT_SIZE[1]-h, :INPUT_SIZE[0]-w]
        parse_array = aug_parse_array

        mask = Image.open(path_mask)
        mask_array = np.array(mask)
        aug_mask_array = np.zeros(mask_array.shape)
        if w < 0:
            aug_mask_array[h:, :INPUT_SIZE[0]+w] = mask_array[:INPUT_SIZE[1]-h, -w:]
        else:
            aug_mask_array[h:, w:] = mask_array[:INPUT_SIZE[1]-h, :INPUT_SIZE[0]-w]
        mask_array = aug_mask_array

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

        mask = (mask_array > 25).astype(np.float32)

        head = torch.from_numpy(head)
        cloth = torch.from_numpy(cloth)
        arms = torch.from_numpy(arms)
        mask = torch.from_numpy(mask)

        crop_cloth = image * cloth + (1 - cloth)
        cloth_ = cloth_ * cloth_mask + (1 - cloth_mask)

        cloth = cloth.unsqueeze_(0)
        mask = mask.unsqueeze_(0)
        arms = arms.unsqueeze_(0)
        
        with open(path_pose, 'rb') as f:
            pose_label = pickle.load(f)
        pose_data = pose_label

        point_num = 18
        pose_map = torch.zeros(point_num, H, W)
        r = self.radius
        pose = Image.new('L', (W, H))
        pose_draw = ImageDraw.Draw(pose)
        for i in range(point_num):
            one_map = Image.new('L', (W, H))
            draw = ImageDraw.Draw(one_map)
            if i in pose_data.keys():
                pointx = pose_data[i][0]+w
                pointy = pose_data[i][1]+h
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
            'pose': pose_map,#pose map
            'name' : name,
            'crop_cloth' : crop_cloth,
            'crop_cloth_mask': cloth,
            'arms_mask': arms,
            'tar_body_mask': mask,
            }
        return result


    def __len__(self):
        return len(self.image_files)

class CFDataLoader(object):
    def __init__(self, opt, dataset):
        super(CFDataLoader, self).__init__()

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

