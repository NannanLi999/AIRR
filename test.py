import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib.pyplot as plt
import torchvision
import argparse
import os

#options: synthesis, attr, celeba, celebahq
DATASET='celeba' 

#deepfashion synthesis
if DATASET=='synthesis':
    from model_synthesis import *
    from dataloader_synthesis import Dataset

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--l1', '--lambda1', default=0.25, type=float, help='lambda for disentanglement')
    parser.add_argument('--l2', '--lambda2', default=0.125, type=float, help='lambda for image attribute')
    parser.add_argument('--l3', '--lambda3', default=1.0, type=float, help='lambda for reconstruction')
    parser.add_argument('--l4', '--lambda4', default=1.0, type=float, help='lambda for perceptual loss')
    parser.add_argument('--lr', '--learning rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', '--beta1', default=0.5, type=float, help='beta1 in Adam')

    save_dir='model/synthesis/'
    classifier_path='classifier/model_synthesis.pth'
    NUM_CLASSES=[17,4]
    NUM_EPOCH=31
    batch_size=16

#deepfashion finegrained attribute    
elif DATASET=='attr':
    from model_attr import *
    from dataloader_attr import Dataset

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--l1', '--lambda1', default=0.05, type=float, help='lambda for disentanglement')
    parser.add_argument('--l2', '--lambda2', default=0.125, type=float, help='lambda for image attribute')
    parser.add_argument('--l3', '--lambda3', default=2.0, type=float, help='lambda for reconstruction')
    parser.add_argument('--l4', '--lambda4', default=1.0, type=float, help='lambda for perceptual loss')
    parser.add_argument('--lr', '--learning rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', '--beta1', default=0.5, type=float, help='beta1 in Adam')

    save_dir='model/attr/'
    classifier_path='classifier/model_attr.pth'
    NUM_CLASSES=[7,3,3,4,6,3]
    NUM_EPOCH=51
    batch_size=16
    
#celeba
elif DATASET=='celeba':
    from model_celeba import *
    from dataloader_celeba import Dataset

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--l1', '--lambda1', default=0.5, type=float, help='lambda for disentanglement')
    parser.add_argument('--l2', '--lambda2', default=0.5, type=float, help='lambda for image attribute')
    parser.add_argument('--l3', '--lambda3', default=1.0, type=float, help='lambda for reconstruction')
    parser.add_argument('--l4', '--lambda4', default=1.0, type=float, help='lambda for perceptual loss')
    parser.add_argument('--lr', '--learning rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', '--beta1', default=0.5, type=float, help='beta1 in Adam')

    save_dir='model/celeba/'
    classifier_path='classifier/model_celeba.pth'
    NUM_CLASSES=[5,3,3,2,2,2,2,2]
    NUM_EPOCH=21
    batch_size=16
    
#celebahq
elif DATASET=='celebahq':
    from model_celebahq import *
    from dataloader_celebahq import Dataset

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--l1', '--lambda1', default=0.25, type=float, help='lambda for disentanglement')
    parser.add_argument('--l2', '--lambda2', default=0.125, type=float, help='lambda for image attribute')
    parser.add_argument('--l3', '--lambda3', default=20.0, type=float, help='lambda for reconstruction')
    parser.add_argument('--l4', '--lambda4', default=10.0, type=float, help='lambda for perceptual loss')
    parser.add_argument('--lr', '--learning rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--beta1', '--beta1', default=0.9, type=float, help='beta1 in Adam')

    save_dir='/model/celebahq/'
    NUM_CLASSES=[5,3,3,2,2,2,2,2]
    NUM_EPOCH=11
    batch_size=4
    
else:
    print('Undefined dataset!')


args = parser.parse_args()
if not os.path.exists(save_dir):
   os.mkdir(save_dir)
model_path=os.path.join(save_dir,'model.pth')

print("==============================")
print("lambda1={},lambda2={},lambda3={},lambda4={}".format(args.l1,args.l2,args.l3,args.l4))
print("==============================")



start_epoch=0

MSELoss=torch.nn.MSELoss(reduction='mean')

def save_img_from_torch(img,imgname,imgfolder='output/'):
   img=np.clip((img+1)/2.0,0,1)
   img=np.transpose(img,[1,2,0])
   name=imgname.replace('.jpg','.png')
   nimg=Image.fromarray(np.uint8(img*255))
   nimg.save(os.path.join(imgfolder,name))
            
def test(device,encoder,generator,discriminator,dataloader,imgfolder='output/'):     
      print("=======test=========")  
      encoder.eval()
      generator.eval()
      discriminator.eval()
      total_loss=0.0
      loss_dict={}
      for ibatch,(imgs,attrs,imgname,paired_attrs,face_seg,paired_imgs,mask) in enumerate(dataloader):
         with torch.no_grad(): 
            mask=mask.to(device) 
            z,_,_=encoder((imgs).to(device),paired_attrs.to(device),mask=mask)           
            gen_x=generator(z)
            gen_x=gen_x*(1-face_seg.to(device))+imgs.to(device)*face_seg.to(device) 
            loss_dict['rec_loss']=MSELoss(gen_x,imgs.to(device))
            
         total_loss+=sum([v.mean() for v in loss_dict.values()])
         if ibatch%50==0:
            index=np.random.choice(np.arange(len(gen_x)),15)
            for ii in index:
              compare=np.concatenate((imgs[ii].cpu().numpy(),gen_x[ii].detach().cpu().numpy()),axis=2)
              save_img_from_torch(compare,imgname[ii],imgfolder=imgfolder)
            loss={k:v.cpu().detach().numpy() for k,v in loss_dict.items()}
            print('total loss=%.2f at img %d'%(total_loss/(ibatch+1),ibatch+1))
            for k,v in loss.items():             
               print('\t%s=%.2f'%(k,np.mean(v)))
                  
def manipulate(device,encoder,generator,discriminator,dataloader,imgfolder='output/'):     
      print("=======manip=========")  
      encoder.eval()
      generator.eval()
      discriminator.eval()
      total_loss=0.0
      loss_dict={}
      plt.figure()
      for ibatch,(imgs,attrs,imgnames,paired_attrs,face_seg,paired_imgs,mask) in enumerate(dataloader):
         with torch.no_grad(): 
            #specify the manipulated attribute here
            attrs[:,1]=0 
            mask=mask.to(device) 
            z,_,_=encoder((imgs).to(device),attrs.to(device),mask=mask)           
            gen_x=generator(z)
            gen_x=gen_x*(1-face_seg.to(device))+imgs.to(device)*face_seg.to(device) 
            loss_dict['rec_loss']=MSELoss(gen_x,imgs.to(device))
        
         total_loss+=sum([v.mean() for v in loss_dict.values()])
         if ibatch%50==0:
            index=np.arange(len(gen_x))
            for ii in index:
              compare=np.concatenate((imgs[ii].cpu().numpy(),gen_x[ii].detach().cpu().numpy()),axis=2)
              save_img_from_torch(compare,imgnames[ii],imgfolder=imgfolder)
            loss={k:v.cpu().detach().numpy() for k,v in loss_dict.items()}
            print('total loss=%.2f at img %d'%(total_loss/(ibatch+1),ibatch+1))
            for k,v in loss.items():             
               print('\t%s=%.2f'%(k,np.mean(v)))
            
     

train_data=Dataset(split='train')
test_data=Dataset(split='test')
val_data=Dataset(split='val')

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False) 
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False) 

device = torch.device('cuda')
encoder=Encoder(ngf=32, num_classes=NUM_CLASSES)
encoder.to(device)
generator=Generator(ngf=32)
generator.to(device)
discriminator=Discriminator(ngf=32, num_classes=NUM_CLASSES)
discriminator.to(device)

enc_params=[par for par in encoder.parameters()] 
gen_params=[par for par in generator.parameters()] 
dis_params=[par for par in discriminator.parameters()] 
optimizerE=torch.optim.Adam(enc_params,lr=args.lr, betas=(args.beta1, 0.999))
optimizerG=torch.optim.Adam(gen_params,lr=args.lr, betas=(args.beta1, 0.999))
optimizerD=torch.optim.Adam(dis_params,lr=args.lr, betas=(args.beta1, 0.999))
print('rough number of parameters:',len(enc_params),len(gen_params),len(dis_params))


if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizerD.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    optimizerE.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    optimizerG.load_state_dict(checkpoint['generator_optimizer_state_dict'])  
    start_epoch=checkpoint['epoch']+1
    batch_size=checkpoint['batch_size']
    test(device,encoder,generator,discriminator,val_dataloader,imgfolder=save_dir)
    
if os.path.exists(model_path):  
   manipulate(device,encoder,generator,discriminator,val_dataloader,imgfolder='manip/')


      