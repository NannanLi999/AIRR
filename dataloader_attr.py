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

LABEL=['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat','Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm','Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']

class Dataset(torch.utils.data.Dataset):#net/ivcfs4/mnt/data
    def __init__(self,root='data/attr/',split='train',cat=0):
         # cat is the category index that you want to manipulate. 
         # options for cat: 0,1,2,3,4,5 
         self.root=root
         self.num_classes=[7,3,3,4,6,3]
         
         self.bg_idx=[LABEL.index(x) for x in ['Background','Hat', 'Hair', 'Glove','Sunglasses', 'Face', 'Left-shoe', 'Right-shoe','Left-arm', 'Right-arm','Left-leg', 'Right-leg']]
         self.face_idx=[LABEL.index(x) for x in ['Hair','Sunglasses', 'Face']]

         self.transform=transforms.ToTensor()
         
         self.img_dir=os.path.join(root,'img_resampled/')
         self.seg_dir=os.path.join(root,'seg/')
                
         self.imgname_list=sorted([x for x in os.listdir(self.img_dir)])
         self.n2n=pickle.load(open(os.path.join(root,'name2pairedname_train.pkl'),'rb'))
       
         index=pickle.load(open(os.path.join(root,'index.pkl'),'rb'))
         self.split=split
         if split=='train':
            self.index=index[:-2000]           
         elif split=='val':
            self.index=index[-2000:-1000] 
         elif split=='test':
            self.index=index[-1000:]
            self.n2n_test=pickle.load(open(os.path.join(root,'name2pairedname_test.pkl'),'rb'))
            self.cat=cat
            self.index=[x for x in index[-1000:] if len(self.n2n_test[self.imgname_list[x]][self.cat])>0]
            
        
         self.attr_dict=pickle.load(open(os.path.join(root,'name2attr.pkl'),'rb'))
         self.attr_str=['floral','graphic','striped','embroidered','pleated','solid','lattice','long_sleeve','short_sleeve','sleeveless','maxi_length','mini_length','no_dress','crew_neckline','v_neckline','square_neckline','no_neckline','denim','chiffon','cotton','leather','faux','knit','tight','loose','conventional']
         
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
            paired_name=self.n2n_test[name][self.cat]
        else:
            paired_name=self.n2n[name]
        
       
        paired_attr=self.attr_dict[paired_name]
        paired_attr=torch.as_tensor(paired_attr,dtype=torch.long)
        
      
        ref_path=os.path.join(self.img_dir,paired_name)      
        paired_img=Image.open(ref_path)
        paired_img=self.transform(paired_img)*2-1
        
        parse=np.array(Image.open(self.seg_dir+name))
        mask=0
        for x in self.bg_idx:
            mask+=np.where(parse==x,1,0)
        mask=1-mask
        if np.sum(mask)<224*224*0.01:
            mask=np.ones(mask.shape)         
        mask=torch.from_numpy(mask[None,:,:].astype(np.float32))   
             
        face_seg=0
        for x in self.face_idx:
            face_seg+=np.where(parse==x,1,0)      
        face_seg=torch.from_numpy(face_seg[None,:,:].astype(np.float32))      
                        
        return img,attr,name,paired_attr,face_seg,paired_img,mask
             
        