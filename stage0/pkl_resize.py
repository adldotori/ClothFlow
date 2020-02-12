import pickle

path_pose = '/home/fashionteam/dataset_MVC_down/3401918/1/pose_resize.pkl'
with open(path_pose, 'rb') as f:
    pose_label = pickle.load(f)
    print(type(pose_label))
    print(pose_label)