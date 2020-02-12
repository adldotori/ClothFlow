import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image

def imsave(nparray,fname):
    Image.fromarray(nparray).save(fname)

TARGET_SIZE = (512,512)
base_dir = "/home/fashionteam/dataset_MVC_bottoms/train/"
folder_list = os.listdir(base_dir)

f = open('delet.txt','w')

t = 0
for i, folder in enumerate(folder_list):
    image_list = os.listdir(os.path.join(base_dir, folder))
    image_list = [image for image in image_list if os.path.isfile(os.path.join(base_dir, folder, image)) and '4x' in image and not ('resize' in image)]
    for image in image_list:            
        I = Image.open(os.path.join(base_dir, folder, image))
        I = np.array(I)
        try:
            img = np.pad(I, ((0,0),(160,160),(0,0)), 'constant', constant_values=(255))
        except:
            print('ERROR', image)
            continue
        img = cv2.resize(img, TARGET_SIZE)
        print(i, image)
        imsave(img, osp.join(base_dir,folder, image.split('.')[0]+'_resize.jpg'))
        
        if os.path.isdir(os.path.join(base_dir, folder, image.split('-')[1])) and image.split('-')[1] in ['p','1','3']:
            # if not (osp.isfile(osp.join(base_dir, folder, image.split('-')[1], 'segment_resize.jpg')) and osp.isfile(osp.join(base_dir, folder, image.split('-')[1], 'segment_vis_resize.jpg'))):
            print(i,image)
            try:
                seg = np.array(Image.open(os.path.join(base_dir, folder, image.split('-')[1], 'segment.png')))
                seg_vis = np.array(Image.open(os.path.join(base_dir, folder, image.split('-')[1], 'segment_vis.png')))
                
                seg = np.pad(seg, ((0,0),(160,160)), 'constant', constant_values=(0))
                seg = cv2.resize(seg, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                seg_vis = np.pad(seg_vis, ((0,0),(160,160),(0,0)), 'constant', constant_values=(0))
                seg_vis = cv2.resize(seg_vis, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                imsave(seg, osp.join(base_dir,folder, image.split('-')[1], 'segment_resize.jpg'))
                imsave(seg_vis, osp.join(base_dir,folder, image.split('-')[1], 'segment_vis_resize.jpg'))
            except OSError:
                f.write(image+'\n')
                print(image)
        else:
            print(folder,image,'There is not dir')

f.close()