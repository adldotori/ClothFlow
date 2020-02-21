import os
from PIL import Image

base_dir = "/home/fashionteam/dataset_MVC_tops/train/"
folder_list = os.listdir(base_dir)

with open('uncomplete.txt','w') as f:
    c = 0
    t = 0
    print(len(folder_list))
    for folder in folder_list:
        image_list = os.listdir(os.path.join(base_dir, folder))
        image_list = [image for image in image_list if os.path.isfile(os.path.join(base_dir, folder, image)) and '4x_resize' in image]
        for image in image_list:
            I = Image.open(os.path.join(base_dir, folder, image))
            if I.size != (1920, 2240) and I.size != (512,512) and I.size != (192, 256):
                print('size unvalid', folder, image)
                print(I.size)
                # print('rm -rf '+'~/dataset_MVC/' + folder)
                # os.system('rm -rf '+'~/NCAP/dataset_MVC/' + folder)
                c+=1
                break
            try:
                detail = os.path.join(base_dir, folder, image.split('-')[1])
                # print(detail)
            except:
                continue 
            if (not (os.path.isfile(detail + '/segment_vis_resize.jpg') and os.path.isfile(detail + '/segment_resize.jpg') and os.path.isfile(detail + '/pose_resize.pkl'))) and image.split('-')[1] in ['p','1','3']:
                f.write('/data/'+folder+'/'+image+'\n')
                t+=1
    print(c)
    print(t)