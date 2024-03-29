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
import time
import torch.nn.functional as F
import random

INPUT_SIZE = (512, 512)

#import time
class timer:
    def __init__(self):
        self.records = {}
        self.lap_records = {}
        self.on()
    def on(self,flag="f1"):
        self.records[flag] = time.time()
        self.lap_records[flag] = self.records[flag]
    def check(self,flag="f1",message=""):
        current = time.time()
        print("(TIMER %s) Total: %fs, lap: %fs - message: %s" %(flag,current-self.records[flag],current-self.lap_records[flag],message))
        self.lap_records[flag] = current




def naming(pair,position):
    return pair + "-" + position + "-4x_resize.jpg"

class CFDataset(data.Dataset):
    """Dataset for CF-VTON.
    """
    def __init__(self, opt, is_tops=True):
        super(CFDataset, self).__init__()
        # base setting
        self.is_tops = is_tops
        self.opt = opt
        self.root = opt.dataroot
        self.root_mask = opt.dataroot_mask
        self.datamode = opt.datamode # train or test or self-defined
        self.stage = opt.stage # GMM or TOM
        self.data_list = opt.data_list
        #print(self.data_list)
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_1ch = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

        self.image_files = os.listdir(self.data_path)
        if self.datamode == 'test':
            self.image_files = os.listdir(self.data_path)[:100]

    def name(self):
        return "CFDataset"

    def __getitem__(self, index):

        name = self.image_files[index]  # the index of the pair
        p = 'p'

        path_image = osp.join(self.data_path, name, naming(name,p)) # condition image path
        image = Image.open(path_image)
        image = self.transform(image)

        # augmentation
        aug_image = torch.zeros(image.shape)
        w = random.randint(-150,150)
        h = random.randint(0,200)
        if w < 0:
            aug_image[:, h:, :INPUT_SIZE[0]+w] = image[:, :INPUT_SIZE[1]-h, -w:]
        else:
            aug_image[:, h:, w:] = image[:, :INPUT_SIZE[1]-h, :INPUT_SIZE[0]-w]
        image = aug_image

        H, W, C = np.array(image).shape###
        # conditionnal parsing and pose path
        path_seg = osp.join(self.data_path, name, p, "segment_resize.png")
        path_pose = osp.join(self.data_path, name, p, "pose_resize.pkl")

        # conditional segment processing
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
        pants = (parse_array == 9).astype(np.float32) + \
                  (parse_array == 12).astype(np.float32)
        face = (parse_array == 13).astype(np.float32)

        face = torch.from_numpy(face)
        with open(path_pose, 'rb') as f:
            pose_label = pickle.load(f)
        pose_data = pose_label

        # rgb_array = np.full((3,INPUT_SIZE[0],INPUT_SIZE[1]), np.random.rand(3))
        # rgb_array = rgb_array.astype(np.float32)
        # rand = torch.from_numpy(rgb_array)

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

        full = image * shape
        lack = image * (shape - head)
        area = torch.mean(face) * INPUT_SIZE[0] * INPUT_SIZE[1]
        size = (torch.sqrt(area)//(torch.rand(1)*0.7+0.8)).int()


        if 0 in pose_data.keys():
            loc_x = pose_data[0][0]+w
            loc_y = pose_data[0][1]+h
            head = image * head + (1 - head) * 0
            tmp = head.clone()
            tmp[:,loc_y-size:loc_y+size,loc_x-size:loc_x+size] = 0
            face = head * np.where(tmp==0,1,0)
            face = face.type(torch.FloatTensor)
            lack = lack + face
        else:
            face = image * head + (1 - head) * 0
            lack = full

        result = {
            'lack': lack,
            'full': full,
            'face': face,
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
    parser.add_argument("--dataroot", default = "/home/fashionteam/NCAP/dataset_MVC_tops")
    parser.add_argument("--datamode", default = "")
    parser.add_argument("--stage", default = "PGP")
    parser.add_argument("--data_list", default = "/home/fashionteam/NCAP/stage1/MVCtops_pair.txt")
    parser.add_argument("--fine_width", type=int, default = 1024)
    parser.add_argument("--fine_height", type=int, default = 1024)
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

