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

class Dataset(torch.utils.data.Dataset):
    def __init__(self,root='data/synthesis/',split='train',cat=0):
         # cat is the category index that you want to manipulate. cat=0 means random category and is used in training.
         self.root=root
         self.num_classes=[17,4]
         self.transform=transforms.ToTensor()

         index=pickle.load(open(os.path.join(root,'index.pkl'),'rb'))
         self.imgname_list=[x[0][0] for x in scio.loadmat(os.path.join(root,'Img/subset_index.mat'))['nameList']]
         self.n2n=pickle.load(open(os.path.join(root,'name2pairedname.pkl'),'rb'))
         
         if split=='train':
            self.index=index[:-4000]         
         elif split=='val':
            self.index=index[-4000:-2000]
         elif split=='test':
            self.index=index[-2000:]           
               
         self.img_root=os.path.join(root,'Img/')
         img_file=h5py.File(os.path.join(root,'G2.h5'),'r')
         self.img_data_minus_mean=img_file['ih']
         self.img_data_mean=np.array(img_file['ih_mean'])
         self.seg_data=np.array(img_file['b_'])
         self.cat=cat
         self.split=split
         
         name2index={x:i for i,x in enumerate(self.imgname_list)}
        
         
         anno_root=os.path.join(root,'Anno/')
         self.anno=scio.loadmat(os.path.join(anno_root,'language_original.mat'))
         self.attr_dict={}
         self.attr_choices=[]
         for i in range(len(self.anno['nameList'])):
           self.attr_dict[self.anno['nameList'][i][0][0]]=[self.anno['color_'][i][0]-1,self.anno['sleeve_'][i][0]-1]
           self.attr_choices.append([self.anno['color_'][i][0]-1,self.anno['sleeve_'][i][0]-1])
           k=str(self.anno['color_'][i][0]-1)+'_'+str(self.anno['sleeve_'][i][0]-1)
          
        
    def __len__(self):
        return len(self.index)
    
        
    def __getitem__(self,idx):
    
        img=np.transpose(np.clip(self.img_data_minus_mean[self.index[idx]]+self.img_data_mean,0,1),[1,2,0])
        img=Image.fromarray(np.uint8(img*255))
        img=self.transform(img)*2-1#torch.from_numpy(img)
        assert (img.min()>=-1) and (img.max()<=1)
        name=self.imgname_list[self.index[idx]]
        
        attr=self.attr_dict[name]
        attr=torch.as_tensor(attr,dtype=torch.long)
        
       
        if self.split=='test':
           paired_name=self.n2n[name][self.cat+1]
        else:                
           paired_name=self.n2n[name][0]
           
        ref_idx=self.imgname_list.index(paired_name)
        paired_img=np.transpose(np.clip(self.img_data_minus_mean[ref_idx]+self.img_data_mean,0,1),[1,2,0])#self.pair_index[idx]
        paired_img=Image.fromarray(np.uint8(paired_img*255))
        paired_img=self.transform(paired_img)*2-1
             
        paired_attr=self.attr_dict[paired_name]
        paired_attr=torch.as_tensor(paired_attr,dtype=torch.long)
        
        l=self.seg_data[self.index[idx]]
        mask=np.where(l==3,1.0,0)
        if mask.sum()==0:
            mask=np.where(l==4,1.0,0)          

        if np.sum(mask)<128*128*0.01:
            mask=np.ones(mask.shape)
            
        mask=torch.from_numpy(mask.astype(np.float32))
             
        face_seg=np.where(l==1,1.0,0)+np.where(l==2,1.0,0)    
        face_seg=torch.from_numpy(face_seg.astype(np.float32))
             
        return img,attr,name,paired_attr,face_seg,paired_img,mask
             