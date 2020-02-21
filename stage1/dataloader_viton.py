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

#import time
INPUT_SIZE = (512, 512)

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
        self.stage = opt.stage # GMM or TOM
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_1ch = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
        
        # load data list
        self.image_files = os.listdir(osp.join(self.data_path,"pose_pkl"))

    def name(self):
        return "CFDataset"

    def __getitem__(self, index):


        name = naming(self.image_files[index])

        # cloth image & cloth mask

        path_cloth = osp.join(self.data_path,"cloth",name+"_1.jpg")
        path_mask = osp.join(self.data_path, "cloth-mask",name+"_1.jpg")
        cloth = Image.open(path_cloth).resize(INPUT_SIZE)
        cloth_mask = Image.open(path_mask).resize(INPUT_SIZE)

        if self.opt.stage == "GMM":
            target_softmax_path = osp.join(self.result_dir+'/PGP', pair+'.jpg')

            if not os.path.exists(target_softmax_path):
                print(target_softmax_path)
            target_softmax_shape = Image.open(target_softmax_path).resize(INPUT_SIZE)
            target_softmax_shape = target_softmax_shape.resize((self.fine_width // 16, self.fine_height // 16), Image.BILINEAR)
            target_softmax_shape = target_softmax_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
            target_softmax_shape = self.transform(target_softmax_shape).type(torch.float32)

        path_image = osp.join(self.data_path, "image",name+"_0.jpg") # condition image path

        image = Image.open(path_image).resize(INPUT_SIZE)

        H, W, C = np.array(image).shape###

        c_mask_array = np.array(cloth_mask)
        c_mask_array = (c_mask_array > 0).astype(np.float32)
        c_mask = torch.from_numpy(c_mask_array)
        cloth_mask = c_mask.unsqueeze_(0)

        original_image = image #

        cloth_ = self.transform(cloth)
        image = self.transform(image)

        # parsing and pose path
        path_seg = osp.join(self.data_path, "image-parse", name+"_0.png")
        path_pose = osp.join(self.data_path, "pose_pkl",name+"_0_keypoints.pkl")

        # segment processing
        seg = Image.open(path_seg).resize(INPUT_SIZE)
        parse_array = np.array(seg)

        parse_fla = parse_array.reshape(H*W, 1)
        parse_fla = torch.from_numpy(parse_fla).long()
        parse = torch.zeros(H*W, 20).scatter_(1, parse_fla, 1)
        parse = parse.view(H, W, 20)
        parse = parse.transpose(2, 0).transpose(2,1).contiguous()  # [20,256,192]

        shape = (parse_array > 0).astype(np.float32)  # condition body shape
        head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 2).astype(np.float32) + \
                 (parse_array == 4).astype(np.float32) + \
                 (parse_array == 13).astype(np.float32)
        cloth = (parse_array == 5).astype(np.float32) + \
                  (parse_array == 6).astype(np.float32) + \
                  (parse_array == 7).astype(np.float32)

        # arms = (parse_array == 15).astype(np.float32) + \
        #           (parse_array == 14).astype(np.float32)
        
        # pants = (parse_array == 9).astype(np.float32) + \
        #           (parse_array == 12).astype(np.float32) + \
        #           (parse_array == 16).astype(np.float32)

        # body_shape = shape
        # shape_array = np.asarray(body_shape)
        # body_fla = shape_array.reshape(H*W,1)
        # body_fla = torch.from_numpy(body_fla).long()
        # one_hot = torch.zeros(H*W, 2).scatter_(1, body_fla, 1)
        # one_hot = one_hot.view(H, W, 2)
        # one_hot = one_hot.transpose(2, 0).transpose(1, 2).contiguous() #[2,256,192]

        # shape = Image.fromarray((shape*255).astype(np.uint8))
        # shape_sample = shape.resize((self.fine_width//16, self.fine_height//16),Image.BILINEAR)
        # shape_sample = shape_sample.resize((self.fine_width, self.fine_height),Image.BILINEAR)


        # cloth_sample = Image.fromarray((cloth*255).astype(np.uint8)) # downsampling of cloth on the person
        # cloth_sample = cloth_sample.resize((self.fine_width//4, self.fine_height//4), Image.BILINEAR)
        # cloth_sample = cloth_sample.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        # shape_sample = self.transform_1ch(shape_sample)

        head = torch.from_numpy(head)
        cloth = torch.from_numpy(cloth)
        # arms = torch.from_numpy(arms)
        # pants = torch.from_numpy(pants)
        # cloth_sample = self.transform_1ch(cloth_sample)
        #c_cloth_sample = self.transform(c_cloth_sample)

        
        crop_head = image * head + (1 - head)
        crop_cloth = image * cloth + (1 - cloth)
        off_cloth = image * (1-cloth) + cloth
        # crop_arms = image * arms + (1-arms)
        # crop_pants = image * pants + (1-pants)



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
        with open(path_pose, 'rb') as f:
            pose_label = pickle.load(f)
        pose_data = pose_label
        ###

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

            #c_pointx = c_pointx * 192 / 762
            #c_pointy = c_pointy * 256 / 1000
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            one_map = self.transform_1ch(one_map)
            pose_map[i] = one_map[0]

        """
        t_pose_data = - np.ones((18, 2), dtype=int)
        with open(path_t_pose, 'rb') as f:
            pose_label = pickle.load(f)
            for i in range(18):
                if pose_label['subset'][0, i] != -1:
                    t_pose_data[i, :] = pose_label['candidate'][int(pose_label['subset'][0, i]), :2]
            t_pose_data = np.asarray(t_pose_data)
        """
        




        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g)
        else:
            im_g = ''
        cloth = cloth.view(1, H, W)


        if self.opt.stage == "GMM":
            result = {
                'cloth': cloth,
                'cloth_mask': cloth_mask,
                'grid_image': im_g,
                'head': crop_head,
                'image': image,
                'pose': pose_map,
                'shape_sample': shape_sample,
            }
            return result
        elif self.opt.stage == "PGP":
            result = {
                'shape_sample': shape_sample,
            }
            return result
        elif self.opt.stage == "Tuning":
            result = {
                'shape_sample': shape_sample,
                'cloth': cloth,
                'cloth_mask': cloth_mask,
                'grid_image': im_g,
                'image': image,
            }
        result = {
            'cloth': cloth_,# original cloth
            'cloth_mask': cloth_mask,# original cloth mask
            'image': image,  # source image
            'head': crop_head,# cropped head from source image
            'pose': pose_map,#pose map
            #'shape': shape,
            # 'shape_sample': shape_sample,
            #'grid_image': im_g,
            # if self.opt.stage == "GMM":
            #     'target_softmax_shape': target_softmax_shape,
            # 'one_hot': one_hot,# one_hot - body shape
            # 'upper_mask': cloth,# cropped cloth mask
            # 'head_mask': head,#head mask
            'crop_cloth': crop_cloth,#cropped cloth
            'crop_cloth_mask' : cloth,#cropped cloth mask
            'name' : name,
            'off_cloth': off_cloth,#source image - cloth
            'parse' : parse,
            # 'cloth_sample': cloth_sample,#coarse cloth mask
            # 'arms_mask': arms,
            # 'pants_mask': pants,
            # 'crop_pants': crop_pants,
            # 'crop_arms': crop_arms,
            }
        #for i in result.keys():
        #    print("%s - type: %s" %(i,type(result[i])))
        # print(t_shape.shape, c_shape.shape, t_shape_mask.shape, im_g.shape)
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

