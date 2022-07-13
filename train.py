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
DATASET='celebahq' 

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
    parser.add_argument('--resume_training', '--resume_training', default=False, type=bool, help='use pretrained model or not')

    save_dir='pretrain/synthesis/'
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
    parser.add_argument('--resume_training', '--resume_training', default=False, type=bool, help='use pretrained model or not')

    save_dir='pretrain/attr/'
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
    parser.add_argument('--resume_training', '--resume_training', default=False, type=bool, help='use pretrained model or not')

    save_dir='pretrain/celeba/'
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
    parser.add_argument('--resume_training', '--resume_training', default=False, type=bool, help='use pretrained model or not')

    save_dir='pretrain/celebahq/'
    NUM_CLASSES=[5,3,3,2,2,2,2,2]
    NUM_EPOCH=11
    batch_size=4
    
else:
    print('Undefined dataset!')


args = parser.parse_args()
if not os.path.exists(save_dir):
   os.mkdir(save_dir)
   
print("==============================")
print("lambda1={},lambda2={},lambda3={},lambda4={}".format(args.l1,args.l2,args.l3,args.l4))
print("==============================")



start_epoch=0

MSELoss=torch.nn.MSELoss(reduction='mean')
SCELoss=nn.CrossEntropyLoss(reduction='mean')
logsoftmax=nn.LogSoftmax(dim=1)

# we don't need a classifier for the perceptual loss in celebahq
if DATASET!='celebahq':
    cnn= torchvision.models.resnet50(pretrained=True)
    cnn.fc=nn.Linear(2048, sum(NUM_CLASSES))
    cnn=cnn.to(torch.device('cuda'))
    checkpoint = torch.load(classifier_path)
    cnn.load_state_dict(checkpoint['model_state_dict'])
    params = list(cnn.parameters())
    for param in params:
        param.requires_grad = False
    weight = params[-2]
    cnn = nn.Sequential(*list(cnn.children())[:-1])

def save_img_from_torch(img,imgname,imgfolder='output/'):
   img=np.clip((img+1)/2.0,0,1)
   img=np.transpose(img,[1,2,0])
   name=imgname.replace('.jpg','.png')
   nimg=Image.fromarray(np.uint8(img*255))
   nimg.save(os.path.join(imgfolder,name))
   

def get_attr_loss(logits,labels):
       cur_index=0
       attr_loss=0
       for i in range(len(NUM_CLASSES)):
           attr_loss+=SCELoss(logits[:,cur_index:cur_index+NUM_CLASSES[i]],labels[:,i])
           cur_index+=NUM_CLASSES[i]
       return attr_loss
       
def get_info_loss(logits,labels):
       cur_index=0
       attr_loss=0
       for i in range(len(NUM_CLASSES)):
           alogits=logits[:,cur_index:cur_index+NUM_CLASSES[i]]           
           aloss,_=torch.max(F.log_softmax(alogits,dim=1),1)
           aloss=torch.maximum(aloss-torch.log(torch.ones_like(aloss)/NUM_CLASSES[i]),1e-2*torch.ones_like(aloss))
           attr_loss+=aloss.mean()
           cur_index+=NUM_CLASSES[i]
       return attr_loss
    
def adv_train(epoch,device,encoder,generator,discriminator,dataloader,optimizerE,optimizerG,optimizerD,paramsE,paramsG,paramsD):
     print("=======adv_train=========") 
     encoder.train()
     generator.train()
     discriminator.train()
     total_loss_dis=0
     total_loss_gen=0
     total_loss=0
     for ibatch,(imgs,attrs,name,paired_attrs,face_seg,paired_imgs,mask) in enumerate(dataloader): #,attrs_onehot,paired_attrs_onehot
           loss_dict={}
          
           for par in paramsD:
              par.requires_grad=False
           for par in paramsG+paramsE:
              par.requires_grad=True
           optimizerE.zero_grad()  

           optimizerG.zero_grad() 
           mask=mask.to(device) 
           if DATASET=='celebahq':
               z,attr_logits_real,attr_logits_fake=encoder(paired_imgs.to(device),attrs.to(device),mask=mask)     
           else:
               z,attr_logits_real,attr_logits_fake=encoder(imgs.to(device),attrs.to(device),mask=mask)     
               
           gen_x=generator(z)   
           # for fashion datasets, directly copy the face
           gen_x=gen_x*(1-face_seg.to(device))+imgs.to(device)*face_seg.to(device)   
           dis_gen_x,gen_logits= discriminator(gen_x)  
           rec_loss=args.l3*torch.abs(gen_x-imgs.to(device)).mean()
           loss_dict['rec_loss']=rec_loss.cpu().item()           
           
           if DATASET=='celebahq':
               z_prime,attr_logits_real,attr_logits_fake=encoder(paired_imgs.to(device),paired_attrs.to(device),mask=mask)
           else:
               z_prime,attr_logits_real,attr_logits_fake=encoder(imgs.to(device),paired_attrs.to(device),mask=mask) 
           gen_x_prime=generator(z_prime) 
           # for fashion datasets, directly copy the face
           gen_x_prime=gen_x_prime*(1-face_seg.to(device))+imgs.to(device)*face_seg.to(device) 
           dis_gen_x_prime,gen_logits_prime= discriminator( gen_x_prime)
           G_cost=0.5*(torch.square(1-dis_gen_x_prime)).mean()+0.5*(torch.square(1-dis_gen_x)).mean()
           loss_dict['generator']=G_cost.cpu().item()                     
            
           attr_loss=args.l2*(get_attr_loss(gen_logits_prime,paired_attrs.to(device))+get_attr_loss(gen_logits,attrs.to(device)))
           loss_dict['attr_fake']=attr_loss.cpu().item()
           
           if DATASET=='celebahq':
                 p_loss=args.l4*(z-paired_imgs.to(device)).norm()/sum(list(z.size()))+args.l4*(z_prime-paired_imgs.to(device)).norm()/sum(list(z.size()))
           else:
                 p_loss=args.l4*torch.abs(cnn(gen_x_prime)-cnn(paired_imgs.to(device))).mean()+torch.abs(cnn(gen_x)-cnn(imgs.to(device))).mean()
           loss_dict['p_loss']=p_loss.cpu().item()
        
           info_loss=get_attr_loss(attr_logits_real,attrs.to(device))+get_info_loss(attr_logits_fake,attrs.to(device))
           info_loss=args.l1*info_loss
           loss_dict['info']=info_loss
          

           (G_cost+rec_loss+attr_loss+info_loss+p_loss).backward()
           
           optimizerE.step()           
           optimizerG.step()
           total_loss_gen+=G_cost.cpu().item()
           #=============discriminator=============
           for par in paramsG+paramsE:
              par.requires_grad=False
           for par in paramsD:
              par.requires_grad=True
           optimizerD.zero_grad()  
           
           dis_x_raw,real_logits= discriminator(imgs.to(device))
           dis_x_raw=torch.square(dis_x_raw-1.0)
           dis_x = dis_x_raw.mean()
           
          
           gen_x=gen_x.detach()
           dis_gen_x,_= discriminator(gen_x)
           dis_gen_x=torch.square(dis_gen_x)
           dis_gen_x = dis_gen_x.mean()
           
           
           gen_x_prime=gen_x_prime.detach()
           dis_gen_x_prime,_= discriminator( gen_x_prime)
           dis_gen_x_prime=torch.square(dis_gen_x_prime)
           dis_gen_x_prime =dis_gen_x_prime.mean()                        
           
           D_cost =0.5*(dis_gen_x+dis_gen_x_prime)+ dis_x
           loss_dict['discriminator']=D_cost.cpu().item()           
           
           attr_loss=2*args.l2*get_attr_loss(real_logits,attrs.to(device))
           loss_dict['attr_real']=attr_loss.cpu().item()
           
           (D_cost+attr_loss).backward()
           
           optimizerD.step()
           total_loss_dis+=D_cost.cpu().item()

           loss_i=sum([v for v in loss_dict.values()])                      
           total_loss+=loss_i
           
           if ibatch%100==0:
              print('total loss=%.3f at img %d epoch %d'%(total_loss/(ibatch+1),(ibatch+1)*batch_size,epoch))
              print('\tG loss=%.3f, D loss=%.3f'%(total_loss_gen/(ibatch+1),total_loss_dis/(ibatch+1)))
              loss={k:v for k,v in loss_dict.items()}
              for k,v in loss.items():
                 print('\t%s=%.3f'%(k,v)) 
              

               
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
            if DATASET=='celebahq':
                z,_,_=encoder((paired_imgs).to(device),paired_attrs.to(device),mask=mask)           
            else:
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
            if DATASET=='celebahq':
                z,_,_=encoder((paired_imgs).to(device),attrs.to(device),mask=mask)           
            else:
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
val_data=Dataset(split='val')

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False) 

device = torch.device('cuda')
encoder=Encoder(ngf=32, num_classes=NUM_CLASSES)
encoder.to(device)
if DATASET=='celebahq':
    generator=MyGenerator(ngf=32)
    generator.to(device)
    generator.init_generator()
else:
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


if args.resume_training:
    checkpoint = torch.load(os.path.join(save_dir,'model.pth'))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizerD.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    optimizerE.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    optimizerG.load_state_dict(checkpoint['generator_optimizer_state_dict'])  
    start_epoch=checkpoint['epoch']+1
    batch_size=checkpoint['batch_size']
    #test(device,encoder,generator,discriminator,val_dataloader,imgfolder='celeb/')
    
#if args.resume_training:   
#   manipulate(device,encoder,generator,discriminator,val_dataloader,imgfolder='manip/')
#   assert 1==0

for e in range(start_epoch,NUM_EPOCH):  
      param_dict={'epoch':e, 'batch_size':batch_size,
               'discriminator_state_dict':discriminator.state_dict(),
               'encoder_state_dict':encoder.state_dict(),
               'generator_state_dict':generator.state_dict(),               
               'encoder_optimizer_state_dict':optimizerE.state_dict(),
               'generator_optimizer_state_dict':optimizerG.state_dict(),
               'decoder_optimizer_state_dict':optimizerD.state_dict()
               }
      adv_train(e,device,encoder,generator,discriminator,train_dataloader,optimizerE,optimizerG,optimizerD,enc_params,gen_params,dis_params)
      if e%5==0:
           torch.save(param_dict,os.path.join(save_dir,'model_%d.pth'%(e)))
           torch.save(param_dict,os.path.join(save_dir,'model.pth'%(e)))
      test(device,encoder,generator,discriminator,val_dataloader,imgfolder=save_dir)
      