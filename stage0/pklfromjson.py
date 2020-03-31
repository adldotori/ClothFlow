import pickle
import json
import argparse
import numpy as np
import os
import os.path as osp
import shutil

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--json', type=str, required=True, help='json file path')
# parser.add_argument('--pkl', type=str, required=True, help='pkl file path')
parser.add_argument('--base_dir', type=str, required=True, help='base directory path')
args = parser.parse_args()

files = os.listdir(osp.join(args.base_dir, 'image'))

for file in files:
    if not osp.isdir(osp.join(args.base_dir,file[:-4])):
        os.makedirs(osp.join(args.base_dir,file[:-4]))
    shutil.copy2(osp.join(args.base_dir, 'image', file), osp.join(args.base_dir, file[:-4], 'image.png'))
    shutil.copy2(osp.join(args.base_dir, 'seg', file[:-4]+'.png'), osp.join(args.base_dir, file[:-4], 'segment.png'))

    with open(osp.join(args.base_dir, 'keypoints', file[:-4]+'_keypoints.json'), 'r') as f:
        json_data = json.load(f)
    openpose_kpt = np.array(json_data['people'][0]['pose_keypoints_2d']).reshape([-1, 3])
    new_dict = dict()
    for i in range(18):
        if openpose_kpt[i,2] > 0.05:
            new_dict[i] = (int(openpose_kpt[i, 0]), int(openpose_kpt[i, 1]))
    with open(osp.join(args.base_dir, file[:-4], 'pose.pkl'), 'wb') as f:
        pickle.dump(new_dict, f)