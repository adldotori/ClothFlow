import os
import random

base_dir = "/home/fashionteam/dataset_MVC_tops/train/"

lidi = os.listdir(base_dir)
f = open("../train_MVCtops_pair.txt", 'wt')
g = open("delete_top.txt",'r')
delete = g.read()
print(delete)
for i in range(len(lidi)):
    tedi = os.listdir(base_dir+lidi[i]+"/")
    rind =  random.randint(0,1)
    rind = 2*rind + 1
    ori = 4 * random.uniform(0,1)
    if lidi[i] in delete:
        print("error3" + lidi[i])
        continue
    if not ("p" in tedi):
        print("error1 "+lidi[i])
        continue
    if ori < 1:
        f.write(lidi[i]+" p p\n")
    if str(rind) in tedi:
        f.write(lidi[i]+" p "+str(rind)+"\n")
    else:
        if not (str(4-rind) in tedi):
            print("error2 "+lidi[i])
            continue
        else:
            f.write(lidi[i]+" p "+str(4-rind)+"\n")
f.close()
