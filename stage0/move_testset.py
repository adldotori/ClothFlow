import os
import os.path as osp
import shutil

base_dir = '/home/fashionteam/viton_512/test/'
target_dir = '/home/fashionteam/underwear_512/'

names = os.listdir(osp.join(base_dir, 'image-mask'))

names = [i[:-6] for i in names][:10]

print(names)

for name in names:
    if not osp.isdir(osp.join(target_dir, name)):
        os.makedirs(osp.join(target_dir, name))
    shutil.copy2(osp.join(base_dir, 'image', name+'_0.jpg'), osp.join(target_dir, name, 'image.png'))
    shutil.copy2(osp.join(base_dir, 'image-parse', name+'_0.png'), osp.join(target_dir, name, 'segment_vis.png'))
    shutil.copy2(osp.join(base_dir, 'image-seg', name+'_0.png'), osp.join(target_dir, name, 'segment.png'))
    shutil.copy2(osp.join(base_dir, 'pose_pkl', name+'_0.pkl'), osp.join(target_dir, name, 'pose.pkl'))
    shutil.copy2(osp.join(base_dir, 'image-mask', name+'_0.png'),
    osp.join(target_dir, name, 'image_mask.jpg'))

    