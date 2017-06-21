import sys
import os
from os import listdir
from os.path import join

# dir = "/home/james/MS_GAN/data/10train_Ext/"
dir = "/Users/zhecanwang/Project/MS_GAN/data/10train/"
folders =os.listdir(dir)
for folder in folders:
    if ".DS_Store" not in folder:
        test=os.listdir(dir + folder)
        for item in test:
            if item.endswith("_2.jpg") or item.endswith("_1.jpg") or item.endswith("_0.jpg") or item.endswith("_3.jpg") or item.endswith("_4.jpg"):
                os.remove(join(dir + folder, item))
