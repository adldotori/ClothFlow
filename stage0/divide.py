import os
import os.path as osp
import shutil
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--json', type=str, required=True, help='json file path')
# parser.add_argument('--pkl', type=str, required=True, help='pkl file path')
parser.add_argument('--base_dir', type=str, required=True, help='base directory path')
args = parser.parse_args()

for i in os.listdir(osp.join(args.base_dir,'image')):
    if not osp.isdir(osp.join(args.base_dir, i[:-4])):
        os.makedirs(osp.join(args.base_dir, i[:-4]))
    shutil.copy2(osp.join(args.base_dir, 'image', i), osp.join(args.base_dir, i[:-4], 'image.png'))

# for i in [i for i in os.listdir(osp.join(args.base_dir)) if i[0] != '.']:
#     for j in os.listdir(osp.join(args.base_dir, i)):
#         os.system('mv '+osp.join(args.base_dir, i, j)+' '+osp.join(args.base_dir, i, 'image.png'))
