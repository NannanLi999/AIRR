import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pickle


from PIL import Image
from torchvision import transforms, utils

class Dataset(data.Dataset):
    def __init__(self, root='/net/ivcfs4/mnt/data/nnli/CelebAMask-HQ/', split='train',cat='Smiling'):
        # cat is the category index that you want to manipulate. 
        self.root=root
        self.split=split
        self.img_dir=root+'CelebA-HQ-img'
        self.transform=transforms.ToTensor()
        
        if cat=='Bald_hair_type':
            cat='Bald'
        
        latent_dir='data/celebahq/celebahq_dlatents_psp.npy'
        dlatents = np.load(latent_dir)
        self.n2n=pickle.load(open('data/celebahq/name2pairedname.pkl','rb'))
        if split=='test':
           self.n2n=pickle.load(open('data/celebahq/name2pairedname_test.pkl','rb'))
        
        train_len = 29000
        self.dlatents = dlatents
        if split=='train':            
            self.index=[x for x in range(train_len)]
        elif split=='test':
            self.index=[int(name) for name in self.n2n[cat].keys()]
        else:
            self.index=[x for x in range(train_len,len(dlatents))]

        self.length = len(self.index)
        self.attr_dict=pickle.load(open('data/celebahq/name2attr.pkl','rb'))
           
        self.cat=cat
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name=str(self.index[idx])
        path=os.path.join(self.img_dir,name+'.jpg')
        img=Image.open(path)
        img=self.transform(img)*2-1
        
        dlatent = torch.tensor(self.dlatents[self.index[idx]])
        attr = torch.tensor(self.attr_dict[name])
        
        if self.split=='test':
            paired_name=self.n2n[self.cat][name]
        else:
            paired_name=self.n2n[name]

        paired_attr=self.attr_dict[paired_name]
        paired_attr=torch.as_tensor(paired_attr,dtype=torch.long)
               
        return dlatent,attr,name,paired_attr,torch.zeros_like(img), img, torch.zeros_like(img)
