import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import scipy.io as scio
import h5py
import pickle
import copy
import random
import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):#/net/ivcfs4/mnt/data
    def __init__(self,root='data/celeba/',split='train',cat=0):
          # cat is the category index that you want to manipulate. cat=-1 means random category and is used in training.
          #options for cat: 0,1,2,3,4,5,6,7
         self.root=root
         self.num_classes=[5,3,3,2,2,2,2,2]
         
         self.transform=transforms.Compose([transforms.CenterCrop(size=178),transforms.Resize(128),transforms.ToTensor()])
         
         self.img_dir=os.path.join(root,'img_align_celeba/img_align_celeba/')
         self.seg_dir=os.path.join(root,'seg/')
                
         self.imgname_list=sorted([x for x in os.listdir(self.img_dir)])
         self.n2n=pickle.load(open(os.path.join(root,'name2pairedname.pkl'),'rb'))
       
         index=pickle.load(open(os.path.join(root,'index.pkl'),'rb'))
         self.split=split
         if split=='train':
            self.index=index[:-2000]           
         elif split=='val':
            self.index=index[-2000:] 
         elif split=='test':
            self.index=index[-2000:]            
            self.cat=cat
            self.index=[x for x in index[-2000:] if len(self.n2n[self.imgname_list[x]][self.cat])>0]
            
        
         self.attr_dict=pickle.load(open(os.path.join(root,'name2attr.pkl'),'rb'))

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self,idx):
    
        name=self.imgname_list[self.index[idx]]
        path=os.path.join(self.img_dir,name)
        img=Image.open(path)
        img=self.transform(img)*2-1#torch.from_numpy(img)
        assert (img.min()>=-1) and (img.max()<=1)
        
        attr=self.attr_dict[name]
        attr=torch.as_tensor(attr,dtype=torch.long)
       
        if self.split=='test':
            paired_name=self.n2n[name][self.cat]
        else:
            paired_name=self.n2n[name][-1]

        paired_attr=self.attr_dict[paired_name]
        paired_attr=torch.as_tensor(paired_attr,dtype=torch.long)
       
        ref_path=os.path.join(self.img_dir,paired_name)      
        paired_img=Image.open(ref_path)
        paired_img=self.transform(paired_img)*2-1
             
        seg=Image.open(os.path.join(self.seg_dir,paired_name.replace('.jpg','.png')) )
        seg=np.array(seg)
        mask=np.where(np.not_equal(seg,0),1,0)[None,:,:]
        if np.sum(mask)<128*128*0.01:
              mask=np.ones(mask.shape)
        mask=torch.as_tensor(mask,dtype=torch.float32)
             
        return img,attr,name,paired_attr,torch.zeros_like(mask),paired_img,mask
        