import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import random

def imsave(nparray,fname):
    Image.fromarray(nparray).save(fname)

TARGET_SIZE = (512,512)
base_dir = "/home/fashionteam/underwear/image/"
target_dir = "/home/fashionteam/underwear_512/"
image_list = os.listdir(base_dir)

if not os.path.isdir(target_dir):
    os.makedirs(target_dir)
# t = 0
# for i, folder in enumerate(folder_list):
# #    image_list = os.listdir(os.path.join(base_dir, folder))
# #    image_list = [image for image in image_list if os.path.isfile(os.path.join(base_dir, folder, image)) and '4x' in image and not ('resize' in image)]
#     image_list = ['cloth.png', 'cloth_mask.png']
#     for image in image_list:            
#         I = Image.open(os.path.join(base_dir, folder, image))
#         I = np.array(I)
        
#         try:
#             if (image == "cloth.png"):
#                 img = np.pad(I, ((0,0),(32,32),(0,0)), 'edge')
#             else:
#                 img = np.pad(I, ((0,0),(32, 32)), "constant", constant_values=(0))		
#         except:
#             print('ERROR', image)
#             continue
        
#         img = cv2.resize(img, TARGET_SIZE)
#         print(i, image)
# #        imsave(img, osp.join(base_dir,folder, image.split('.')[0]+'_resize.jpg'))
#         imsave(img, osp.join('/home/fashionteam/viton_re/', folder, image))

        # if os.path.isdir(os.path.join(base_dir, folder, image.split('-')[1])) and image.split('-')[1] in ['p','1','3']:
        #     # if not (osp.isfile(osp.join(base_dir, folder, image.split('-')[1], 'segment_resize.jpg')) and osp.isfile(osp.join(base_dir, folder, image.split('-')[1], 'segment_vis_resize.jpg'))):
        #     print(i,image)
        #     try:
        #         seg = np.array(Image.open(os.path.join(base_dir, folder, image.split('-')[1], 'segment.png')))
        #         seg_vis = np.array(Image.open(os.path.join(base_dir, folder, image.split('-')[1], 'segment_vis.png')))
                
        #         seg = np.pad(seg, ((0,0),(160,160)), 'constant', constant_values=(0))
        #         seg = cv2.resize(seg, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        #         seg_vis = np.pad(seg_vis, ((0,0),(160,160),(0,0)), 'constant', constant_values=(0))
        #         seg_vis = cv2.resize(seg_vis, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        #         imsave(seg, osp.join(base_dir,folder, image.split('-')[1], 'segment_resize.jpg'))
        #         imsave(seg_vis, osp.join(base_dir,folder, image.split('-')[1], 'segment_vis_resize.jpg'))
        #     except OSError:
        #         f.write(image+'\n')
        #         print(image)
        # else:
        #     print(folder,image,'There is not dir')

# os.makedirs('/home/fashionteam/viton_512/train/cloth-mask_/')
for i, image in enumerate(image_list):
#    image_list = os.listdir(os.path.join(base_dir, folder))
#    image_list = [image for image in image_list if os.path.isfile(os.path.join(base_dir, folder, image)) and '4x' in image and not ('resize' in image)]

    I = Image.open(os.path.join(base_dir, image))
    I = np.array(I)
    
    try:
        img = np.pad(I, ((0,0),(32,32),(0,0)), mode='edge')
    except:
        print('ERROR', image)
        continue
    
    # img = cv2.resize(img, TARGET_SIZE)
    print(i, image)
#        imsave(img, osp.join(base_dir,folder, image.split('.')[0]+'_resize.jpg'))
    try:
        os.makedirs(osp.join(target_dir, image[:-4]))
    except:
        pass
    imsave(img, osp.join(target_dir, image[:-4], 'image.png'))