import pickle
import os

def pickle_load(name="opt.pkl"):
    with open(name,"rb") as f:
        d = pickle.load(f)
    return d

stage1 = '/home/fashionteam/ClothFlow/stage1/viton_stage1.pkl'
all_s = '/home/fashionteam/ClothFlow/viton_real_train.pkl'

stage1 = pickle_load(stage1)
all_s = pickle_load(all_s)

print(len(stage1))
print(len(all_s))

print(len([i for i in stage1 if i in all_s]))
print([i for i in stage1 if i in all_s])
with open('/home/fashionteam/ClothFlow/stage1/stage1_dat.pkl', 'wb') as fp:
    pickle.dump([i for i in stage1 if i in all_s], fp)