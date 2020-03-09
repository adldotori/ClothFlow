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
        self.root_cloth = opt.dataroot_cloth
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
        self.result_dir = opt.result_dir

        # load data list
        pairs = []
        c_names = []
        t_names = []

        with open('../'+self.data_list, 'r') as f:
            for line in f.readlines():
                pair, c_name, t_name = line.strip().split()
                pairs.append(pair)
                c_names.append(c_name)
                t_names.append(t_name)

        self.pairs = pairs
        self.c_names = c_names
        self.t_names = t_names

    def name(self):
        return "CFDataset"

    def __getitem__(self, index):

        pair = self.pairs[index]  # the index of the pair
        c_name = self.c_names[index]  # condition person index
        t_name = self.t_names[index]  # target person index

        # cloth image & cloth mask
        path_cloth = osp.join(self.data_path,pair,"crop.png")
        path_mask = osp.join(self.data_path, pair,"mask.png")
        cloth = Image.open(path_cloth).resize(INPUT_SIZE)
        cloth_mask = Image.open(path_mask).resize(INPUT_SIZE)

        path_c_image = osp.join(self.data_path, pair, naming(pair,c_name)) # condition image path
        path_t_image = osp.join(self.data_path, pair, naming(pair,t_name)) # target image path
        c_image = Image.open(path_c_image).resize(INPUT_SIZE)
        t_image = Image.open(path_t_image).resize(INPUT_SIZE)

        H, W, C = np.array(c_image).shape###

        c_mask_array = np.array(cloth_mask)
        c_mask_array = (c_mask_array > 0).astype(np.float32)
        c_mask = torch.from_numpy(c_mask_array)
        cloth_mask = c_mask.unsqueeze_(0)

        warped = Image.open(osp.join(self.root_cloth,pair+".jpg"))
        # warped_mask = Image.open(osp.join(self.root_mask,pair+".jpg"))
        # warped_mask = (np.array(warped_mask) > 0).astype(np.float32)

        cloth = self.transform(cloth)
        c_image = self.transform(c_image)
        t_image = self.transform(t_image)
        warped = self.transform(warped)
        
        # conditionnal parsing and pose path
        path_c_seg = osp.join(self.data_path, pair, c_name, "segment_resize.png")
        if self.is_tops:
            path_c_pose = osp.join(self.data_path, pair, c_name, "pose_resize.pkl")
        #target parsing and pose path
        t_name = t_name
        path_t_seg = osp.join(self.data_path, pair, t_name,  "segment_resize.png")
        path_t_vis_seg = osp.join(self.data_path, pair, t_name,  "segment_vis_resize.png")
        if self.is_tops:
            path_t_pose = osp.join(self.data_path, pair, t_name, "pose_resize.pkl")

        # conditional segment processing

        c_seg = Image.open(path_c_seg).resize(INPUT_SIZE)
        c_parse_array = np.array(c_seg)

        
        
        #c_parse_fla = c_parse_array.reshape(H*W, 1)
        #c_parse_fla = torch.from_numpy(c_parse_fla).long()
        #c_parse = torch.zeros(H*W, 30).scatter_(1, c_parse_fla, 1)
        #c_parse = c_parse.view(H, W, 30)

        #c_parse = c_parse.transpose(2, 0).transpose(2,1).contiguous()  # [20,256,192]



        c_shape = (c_parse_array > 0).astype(np.float32)  # condition body shape

        """
        
        c_head = (c_parse_array == 1).astype(np.float32) + \
                 (c_parse_array == 2).astype(np.float32) + \
                 (c_parse_array == 4).astype(np.float32) + \
                 (c_parse_array == 13).astype(np.float32)
        """
        
        c_shape = (c_parse_array > 0).astype(np.float32)  # condition body shape
        c_cloth = (c_parse_array == 5).astype(np.float32) + \
                  (c_parse_array == 6).astype(np.float32) + \
                  (c_parse_array == 7).astype(np.float32)

        c_body_shape = c_shape
        c_shape_array = np.asarray(c_body_shape)
        c_body_fla = c_shape_array.reshape(H*W,1)
        c_body_fla = torch.from_numpy(c_body_fla).long()
        one_hot_c = torch.zeros(H*W, 2).scatter_(1, c_body_fla, 1)
        one_hot_c = one_hot_c.view(H, W, 2)
        one_hot_c = one_hot_c.transpose(2, 0).transpose(1, 2).contiguous()

        c_shape = Image.fromarray((c_shape*255).astype(np.uint8))
        c_shape_sample = c_shape.resize((self.fine_width//16, self.fine_height//16),Image.BILINEAR)
        c_shape_sample = c_shape_sample.resize((self.fine_width, self.fine_height),Image.BILINEAR)




        c_cloth_sample = Image.fromarray((c_cloth*255).astype(np.uint8)) # downsampling of cloth on the person
        c_cloth_sample = c_cloth_sample.resize((self.fine_width//4, self.fine_height//4), Image.BILINEAR)
        c_cloth_sample = c_cloth_sample.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        c_shape_sample = self.transform_1ch(c_shape_sample)




        #c_head = torch.from_numpy(c_head)
        c_cloth = torch.from_numpy(c_cloth)
        c_cloth_sample = self.transform_1ch(c_cloth_sample)
        #c_cloth_sample = self.transform(c_cloth_sample)

        t_seg = Image.open(path_t_seg).resize(INPUT_SIZE)
        t_parse_array = np.array(t_seg)
        t_vis_seg = self.transform(Image.open(path_t_vis_seg).resize(INPUT_SIZE))
        # t_parse_fla = t_parse_array.reshape(H*W, 1)
        t_parse_fla = t_parse_array.reshape(H*W, 1)
        t_parse_fla = torch.from_numpy(t_parse_fla).long()
        t_parse = torch.zeros(H*W, 30).scatter_(1, t_parse_fla, 1)
        t_parse = t_parse.view(H, W, 30)
        t_parse = t_parse.transpose(2, 0).transpose(1, 2).contiguous()
        
        t_shape = (t_parse_array > 0).astype(np.float32)  # condition body shape
        t_head = (t_parse_array == 1).astype(np.float32) + \
                 (t_parse_array == 2).astype(np.float32) + \
                 (t_parse_array == 4).astype(np.float32) + \
                 (t_parse_array == 13).astype(np.float32)
        t_cloth = (t_parse_array == 5).astype(np.float32) + \
                  (t_parse_array == 6).astype(np.float32) + \
                  (t_parse_array == 7).astype(np.float32)

        t_arms = (t_parse_array == 15).astype(np.float32) + \
                  (t_parse_array == 14).astype(np.float32)
        
        t_pants = (t_parse_array == 9).astype(np.float32) + \
                    (t_parse_array == 12).astype(np.float32)

        # to obtain the one hot of t_shape, the value of each pixel is 1 or 0
        t_body_parse = t_shape
        # #t_body_parse = self.transform_1ch(t_body_parse)
        t_body_parse = torch.from_numpy(t_body_parse)

        head = torch.from_numpy(t_head)
        crop_cloth = torch.from_numpy(t_cloth)
        # warped_mask = torch.from_numpy(warped_mask)
        arms = torch.from_numpy(t_arms)
        pants = torch.from_numpy(t_pants)
        #shape = torch.from_numpy(shape)
        # cloth_sample = self.transform_1ch(cloth_sample)
        #c_cloth_sample = self.transform(c_cloth_sample)

        
        crop_head = t_image * head + (1 - head)
        crop_cloth = t_image * crop_cloth + (1 - crop_cloth)
        off_cloth = t_image * (1-warped_mask) + warped_mask
        crop_arms = t_image * arms + (1-arms)
        crop_pants = t_image * pants + (1-pants)
        #print(t_body_parse.shape)
        
        t_body_parse = t_body_parse.view((1,self.fine_height,self.fine_width))
        
        pants = pants.view(1, H, W)
        """
        t_shape_array = np.asarray(t_body_parse)
        t_body_fla = t_shape_array.reshape(H*W, 1)
        t_body_fla = torch.from_numpy(t_body_fla).long()
        one_hot_t = torch.zeros(H*W, 2).scatter_(1, t_body_fla, 1)
        one_hot_t = one_hot_t.view(H, W, 2)
        one_hot_t = one_hot_t.transpose(2, 0).transpose(1, 2).contiguous()  # [2,256,192]


        t_shape_mask = Image.fromarray((t_shape_mask * 255).astype(np.uint8))
        t_shape = self.transform_1ch(t_shape_mask)
        t_shape_mask = t_shape_mask.resize((self.fine_width // 16, self.fine_height // 16), Image.BILINEAR)
        t_shape_mask = t_shape_mask.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        t_shape_mask = self.transform_1ch(t_shape_mask)
        t_head_mask = torch.from_numpy(t_head_mask)
        t_cloth = torch.from_numpy(t_cloth_mask)
        con_head = c_image * c_head - (1 - c_head)
        con_cloth = c_image * c_cloth + (1 - c_cloth)

        tar_head = t_image * t_head_mask - (1 - t_head_mask)
        tar_cloth = t_image * t_cloth + (1 - t_cloth)

        """
        
        """
        c_pose_data = - np.ones((18, 2), dtype=int)
        with open(path_c_pose, 'rb') as f:
            pose_label = pickle.load(f)
            for i in range(18):
                if pose_label['subset'][0, i] != -1:
                    c_pose_data[i, :] = pose_label['candidate'][int(pose_label['subset'][0, i]), :2]
            c_pose_data = np.asarray(c_pose_data)
        """
        #start_time = time.time()
        ###

        if self.is_tops:
            with open(path_c_pose, 'rb') as f:
                pose_label = pickle.load(f)
            c_pose_data = pose_label
            ###

            point_num = 18
            c_pose_map = torch.zeros(point_num, H, W)
            r = self.radius
            c_pose = Image.new('L', (W, H))
            c_pose_draw = ImageDraw.Draw(c_pose)
            for i in range(point_num):
                one_map = Image.new('L', (W, H))
                draw = ImageDraw.Draw(one_map)
                if i in c_pose_data.keys():
                    c_pointx = c_pose_data[i][0]
                    c_pointy = c_pose_data[i][1]
                else:
                    c_pointx = -1
                    c_pointy = -1

                #c_pointx = c_pointx * 192 / 762
                #c_pointy = c_pointy * 256 / 1000
                if c_pointx > 1 and c_pointy > 1:
                    draw.rectangle((c_pointx - r, c_pointy - r, c_pointx + r, c_pointy + r), 'white', 'white')
                    c_pose_draw.rectangle((c_pointx - r, c_pointy - r, c_pointx + r, c_pointy + r), 'white', 'white')
                one_map = self.transform_1ch(one_map)
                c_pose_map[i] = one_map[0]
            c_pose_s = self.transform_1ch(c_pose)
            """
            t_pose_data = - np.ones((18, 2), dtype=int)
            with open(path_t_pose, 'rb') as f:
                pose_label = pickle.load(f)
                for i in range(18):
                    if pose_label['subset'][0, i] != -1:
                        t_pose_data[i, :] = pose_label['candidate'][int(pose_label['subset'][0, i]), :2]
                t_pose_data = np.asarray(t_pose_data)
            """
            ###
            with open(path_t_pose, 'rb') as f:
                pose_label = pickle.load(f)
            t_pose_data = pose_label
        ###

            point_num = 18
            t_pose_map = torch.zeros(point_num, H, W)
            r = self.radius
            t_pose = Image.new('L', (W, H))
            t_pose_draw = ImageDraw.Draw(t_pose)
            for i in range(point_num):
                one_map = Image.new('L', (W, H))
                draw = ImageDraw.Draw(one_map)
                if i in t_pose_data.keys():
                    t_pointx = t_pose_data[i][0]
                    t_pointy = t_pose_data[i][1]
                else:
                    t_pointx = -1
                    t_pointy = -1
                if t_pointx > 1 and t_pointy > 1:
                    draw.rectangle((t_pointx - r, t_pointy - r, t_pointx + r, t_pointy + r), 'white', 'white')
                    t_pose_draw.rectangle((t_pointx - r, t_pointy - r, t_pointx + r, t_pointy + r), 'white', 'white')
                one_map = self.transform_1ch(one_map)
                t_pose_map[i] = one_map[0]
            t_pose_s = self.transform_1ch(t_pose)

        if self.is_tops:
            target_pose_png_path = osp.join(self.data_path, pair, t_name, "pose.png")
            target_pose_png = np.array(Image.open(target_pose_png_path).resize(INPUT_SIZE))
            target_pose_png = np.expand_dims(target_pose_png, 2)
            target_pose_png = self.transform_1ch(target_pose_png)
        #print(time.time()-start_time)
        # just for visualization
        if self.is_tops:
            c_v_pose = self.transform_1ch(c_pose)
        if self.is_tops:
            t_v_pose = self.transform_1ch(t_pose)

        # cloth-agnostic representation
        c_shape = self.transform_1ch(c_shape)
        #agnostic = torch.cat([c_shape, t_pose_map], 0)
        #agnostic_sample = torch.cat([c_shape_sample, t_pose_map], 0)

        if self.is_tops:
            result = {
                'cloth': cloth,
                'cloth_mask': cloth_mask,# original cloth mask
                'image': t_image,  # source image
                'head': crop_head,# cropped head from source image
                'pose': t_pose_map,#pose map
                'shape': t_shape,
                'upper_mask': cloth,# cropped cloth mask
                'crop_cloth': crop_cloth,#cropped cloth
                'tar_mask' : cloth,#cropped cloth mask
                'name' : pair,
                'warped' : warped,
                'off_cloth': off_cloth,#source image - cloth
                'parse' : t_parse,
                'pants_mask': pants,
                'crop_pants': crop_pants,
                'crop_arms': crop_arms,
                't_name': t_name,
                }
        else:
            result = {
                'cloth': cloth,
                'cloth_mask': cloth_mask,
                'image': t_image,  # source image\
                'crop_cloth_mask': t_pants,
                'warped' : warped,
                'off_cloth': off_cloth,#source image - cloth
                'parse' : t_parse,
                'crop_pants': crop_pants,
                'name': pair,
                'target_body_shape': t_body_parse,
                't_name': t_name,
                'tar_mask': pants,
                }
        
        return result


    def __len__(self):
        return len(self.pairs)

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

