import os
from PIL import Image

base_dir = "/home/fashionteam/dataset_MVC_bottoms/train/"
folder_list = os.listdir(base_dir)

with open('delete_bot.txt','w') as f:
    c = 0
    t = 0
    fs = 0
    a = 0
    print(len(folder_list))
    for folder in folder_list:
        image_list = os.listdir(os.path.join(base_dir, folder))
        image_list = [image for image in image_list if os.path.isfile(os.path.join(base_dir, folder, image)) and '4x_resize' in image]
        if len(image_list) <= 1:
            fs += 1
            f.write(folder+'\n')
        for image in image_list:
            a+=1
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
            if (not (os.path.isfile(os.path.join(base_dir, folder,'crop.png')) and os.path.isfile(detail + '/segment_vis_resize.png') and os.path.isfile(detail + '/segment_resize.png') and os.path.isfile(detail + '/pose_resize.pkl'))) and image.split('-')[1] in ['p','1','3']:
                print(os.path.join(base_dir, folder,'crop.png'))
                f.write('/data/'+folder+'/'+image+'\n')
                t+=1
    print(c)
    print(t)
    print(fs)
    print(a)