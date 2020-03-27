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
#import time
INPUT_SIZE = (512, 512)

def load_pkl(name):
    with open(name,"rb") as f:
        data = pickle.load(f)
    return data

def save_pkl(path,data):
    with open(path, "wb") as f:
        pickle.dump(data,f)

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
        self.image_files = os.listdir(osp.join(self.root, 'cloth'))

    def name(self):
        return "CFDataset"

    def get_result(self):
        return self.result

    def __getitem__(self, index):

        name = self.image_files[index][:-6]
        

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

        low_head = 0
        high_cloth = 0
        for i in range(INPUT_SIZE[0]):
            if np.mean(cloth[i]) > 0 and high_cloth == 0:
                high_cloth = i
            if np.mean(head[i]) > 0:
                low_head = i

        if high_cloth >= low_head:
            return name
        else:
            return []


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
    parser.add_argument("--dataroot", default = "/home/fashionteam/viton_512/train")
    parser.add_argument("--datamode", default = "")
    parser.add_argument("--stage", default = "PGP")
    parser.add_argument("--data_list", default = "MVCup_pair.txt")
    parser.add_argument("--fine_width", type=int, default = 512)
    parser.add_argument("--fine_height", type=int, default = 512)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    
    opt = parser.parse_args()
    dataset = CFDataset(opt)
    data_loader = CFDataLoader(opt, dataset)
    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))

    result = []
    for step in range(len(dataset)):
        inputs = data_loader.next_batch()
        result+=inputs
    print(result)
    save_pkl('viton_stage1.pkl',result)
