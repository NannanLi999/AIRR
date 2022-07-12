import numpy as np
from PIL import Image
import pickle
import os
import re
import matplotlib.pyplot as plt
import copy

imgf=['%s.txt'%x for x in ('train','val','test')]
root='data/attr/'
savedir='data/attr/img_resampled/'

all_names=[]
for bf in imgf:
  with open(os.path.join(root,bf),'r') as f:
     for line in f:
        name='-'.join(line.strip().split('/')[1:]).replace('.jpg','.png')
        all_names.append(name)
        
for name in all_names:
    rname=name.replace('.png','.jpg').replace('-img','/img')
    img=Image.open(os.path.join(root,'img/'+rname))
    width, height = img.size
    side=max(height,width)
    nimg=255*np.ones((side,side,3),dtype=np.uint8)
    nimg[(side-height)//2:height+(side-height)//2,(side-width)//2:width+(side-width)//2]=np.array(img)
    nimg=Image.fromarray(nimg)
    #assert np.array(img).shape==(300,300,3),print(np.array(img).shape)
    nimg=nimg.resize((224,224),Image.BILINEAR)
    nimg.save(os.path.join(savedir,name))