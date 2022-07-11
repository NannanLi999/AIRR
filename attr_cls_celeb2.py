import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from dataloader_celeb2 import CelebaAttrDataset,CelebaDataset
from model_stylegan import *
from PIL import Image
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision
import pickle
import argparse

batch_size=2
start_epoch=0

SCELoss=nn.CrossEntropyLoss(reduction='none')
num_classes=[5,3,3,2,2,2,2,2]

class LCNet(nn.Module):
    def __init__(self, fmaps=[9216, 2048, 512, 21], activ='leakyrelu'):
        super().__init__()
        # Linear layers
        self.fcs = nn.ModuleList()
        for i in range(len(fmaps)-1):
            in_channel = fmaps[i]
            out_channel = fmaps[i+1]
            self.fcs.append(nn.Linear(in_channel, out_channel, bias=True))
        # Activation
        if activ == 'relu':
            self.relu = nn.ReLU()
        elif activ == 'leakyrelu':
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = self.relu(layer(x))
        y =self.fcs[-1](x)
        return x,y


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', type=str, default='step',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=20,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--turning-point', type=int, default=100,
                    help='epoch number from linear to exponential decay mode')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
args = parser.parse_args()

def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']
    """Sets the learning rate to the initial LR decayed by 10 following schedule"""
    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** (epoch // args.step))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * epoch / args.epochs)) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - epoch / args.epochs)
    elif args.lr_decay == 'linear2exp':
        if epoch < args.turning_point + 1:
            # learning rate decay as 95% at the turning point (1 / 95% = 1.0526)
            lr = args.lr * (1 - epoch / int(args.turning_point * 1.0526))
        else:
            lr *= args.gamma
    elif args.lr_decay == 'schedule':
        if epoch in args.schedule:
            lr *= args.gamma
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_img_from_torch(img,imgname,img_mean,imgfolder='output/'):
   img=np.clip((img+1)/2.0,0,1)#np.clip(img+img_mean,0,1)#
   img=np.transpose(img,[1,2,0])
   name="_".join(imgname.split('/')[-2:]).split('.')[0]+'.png'
   nimg=Image.fromarray(np.uint8(img*255))
   nimg.save(imgfolder+name)
        
def get_attr_loss(logits,labels,mask):
       cur_index=0
       attr_loss=0
       for i in range(len(num_classes)):
           attr_loss+=mask[:,i]*SCELoss(logits[:,cur_index:cur_index+num_classes[i]],labels[:,i])#           
           cur_index+=num_classes[i]
       attr_loss=attr_loss.sum()/mask.sum()
       return attr_loss
       
def get_correct_pred_cnt(logits,labels,mask):    
     cur_index=0
     correct=[]
     for i in range(len(num_classes)):
         pred=torch.argmax(logits[:,cur_index:cur_index+num_classes[i]],dim=1)#
         c=mask[:,i]*torch.eq(pred,labels[:,i])
         correct.append(c.to(dtype=torch.float32).sum().cpu().item())
         cur_index+=num_classes[i]
     return np.array(correct)
     
def get_pred(logits):    
     cur_index=0
     preds=[]
     for i in range(len(num_classes)):
         pred=torch.argmax(logits[:,cur_index:cur_index+num_classes[i]],dim=1)#
         preds.append(pred)
         cur_index+=num_classes[i]
     return torch.stack(preds,dim=1)

def train(epoch,device,model,dataloader,optimizer,params):
     print("=======adv_train=========") 
     model.train()
     total_loss=0
     for ibatch,(imgs,attrs,mask,names) in enumerate(dataloader): 
           
           optimizer.zero_grad()  
           logits=model(imgs.to(device))
           loss=get_attr_loss(logits,attrs.to(device),mask.to(device))
           loss.backward()
           optimizer.step()
           total_loss+=loss.cpu().item()
           if ibatch%100==0:
              print('total loss=%.3f at img %d epoch %d'%(total_loss/(ibatch+1),(ibatch+1)*batch_size,epoch))

               
def test(device,model,dataloader):     
      print("=======test=========")  
      model.eval()      
      total_loss=0.0
      correct=np.zeros(len(num_classes))
      total_cnt=0
      for ibatch,(imgs,attrs,mask,names) in enumerate(dataloader):
         with torch.no_grad(): 
           logits=model(imgs.to(device))
           loss=get_attr_loss(logits,attrs.to(device),mask.to(device))
           total_loss+=loss.cpu().item()
           correct+=get_correct_pred_cnt(logits,attrs.to(device),mask.to(device))
           total_cnt+=mask.sum(dim=0).cpu().numpy()
         if ibatch%100==0:
              print('total loss=%.3f at img %d'%(total_loss/(ibatch+1),(ibatch+1)*batch_size))
              print('accu=',correct/(1e-6+total_cnt))
      print('accu=',100*correct/total_cnt,np.mean(100*correct/total_cnt))
      
def fill(device,model,dataloader):     
      print("=======test=========")  
      model.eval()      
      total_loss=0.0
      correct=np.zeros(len(num_classes))
      total_cnt=0
      pseudo_attr={}
      for ibatch,(imgs,attrs,mask,names) in enumerate(dataloader):
         with torch.no_grad(): 
           logits=model(imgs.to(device))
           loss=get_attr_loss(logits,attrs.to(device),mask.to(device))
           total_loss+=loss.cpu().item()
           correct+=get_correct_pred_cnt(logits,attrs.to(device),mask.to(device))
           total_cnt+=mask.sum(dim=0).cpu().numpy()
         if ibatch%100==0:
              print('total loss=%.3f at img %d'%(total_loss/(ibatch+1),(ibatch+1)*batch_size))
              print('accu=',correct/(1e-6+total_cnt))
         preds=get_pred(logits)
         pseudo_label=torch.where(torch.eq(mask.to(device),0),preds,attrs.to(device)).cpu().numpy()
         for i,name in enumerate(names):
             pseudo_attr[name]=pseudo_label[i]
      pickle.dump(pseudo_attr,open('/net/ivcfs4/mnt/data/nnli/CelebaHQ/name2attr.pkl','wb'))

def save_img_from_torch(img,imgname,imgfolder='output/'):
   img=np.clip((img+1)/2.0,0,1)#np.clip(img+img_mean,0,1)#
   img=np.transpose(img,[1,2,0])
   name=imgname+'.png'
   nimg=Image.fromarray(np.uint8(img*255))
   nimg.save(imgfolder+name)
   
def get_gallery(model):
    model.eval() 
    data=np.load('../Dis/lt/data/celebahq_dlatents_psp.npy')
    for k in range(30000):
            w_0 = data[k][None,:,:]
            w_0 = torch.tensor(w_0).to(device)
            with torch.no_grad():
                feats,_= model(w_0.view(w_0.size(0), -1))
                if k%1000==0:
                    print(k,feats.cpu().numpy().shape)
            np.save('resnet_features/celeb2/'+str(k)+'.npy',feats.cpu().numpy())
            
def get_feats():
      feats=[]
      for k in range(30000):
           feats.append(np.load('resnet_features/celeb2/'+str(k)+'.npy'))
      return np.concatenate(feats,axis=0)
      
def to_label(name2attr):
   data_gan=CelebaDataset(split='test') 
   label=[]
   for k in range(30000):
      label.append(name2attr[str(k)])
   label=np.stack(label,axis=0)
   return label
      
import sys 
sys.path.append('pixel2style2pixel/')

from argparse import Namespace
from pixel2style2pixel.options.test_options import TestOptions
from pixel2style2pixel.models.psp import pSp
    
def retrieve(device,encoder,generator,model,cat,net,str2catid,catid2num,all_feats=None,all_label=None):
      print("=======retrieve-gan=========")  
      data_gan=CelebaDataset(split='test',cat=cat) 
      dataloader_gan = DataLoader(data_gan, batch_size=2, shuffle=False)
      model.eval()  
      encoder.eval()
      generator.eval()  
      
      if all_label is None:
        all_feats=get_feats()
        all_label=to_label(data_gan.attr_dict)
      tnorm=np.linalg.norm(all_feats,ord=2,axis=1)              
      R_THRESH=[5,20]
      hit={x:0 for x in R_THRESH}
      for ibatch,(dlatent,attrs,imgname,paired_attrs,imgs) in enumerate(dataloader_gan):
         with torch.no_grad(): 
           
            z,_,_=encoder((dlatent).to(device),paired_attrs.to(device))        
            gen_x_1024=generator(z)
            
            gen_x=F.interpolate(gen_x_1024, size=(256,256))
            _,gen_latents=net(gen_x, randomize_noise=False, resize=opts.resize_outputs, return_latents=True)
            cur_feats,_=model(gen_latents.view(-1, 18*512))
            cur_feats=cur_feats.cpu().numpy()
            paired_attrs=paired_attrs.cpu().numpy()
         for i in range(len(cur_feats)):
               feats=cur_feats[i]
               sim=np.sum(feats[None,:]*all_feats,axis=1)/(np.linalg.norm(feats,ord=2)*tnorm)
               sim=np.abs(sim)
               sorted_sim=np.argsort(-sim)
               for th in R_THRESH:
                  indexes=sorted_sim[:th]
                  cand_attrs=all_label[indexes]
                  t=np.equal(cand_attrs,paired_attrs[i])
                  if len(num_classes) in np.sum(t,axis=1):
                    hit[th]+=1
         if ibatch%10==0:
                print(ibatch,{h:hit[h]/((ibatch+1)*2) for h in hit.keys()})     
                #assert 1==0           
      print({h:hit[h]/catid2num[str2catid[cat]] for h in hit.keys()})#len(data_gan.index)
      r={h:hit[h]/catid2num[str2catid[cat]] for h in hit.keys()} 
      return r,all_feats,all_label
            
def test_gan(device,encoder,generator,model,cat,net,str2catid,catid2num):     
      print("=======test-gan=========")  
      data_gan=CelebaDataset(split='test',cat=cat) 
      dataloader_gan = DataLoader(data_gan, batch_size=2, shuffle=False)
      model.eval()  
      encoder.eval()
      generator.eval()    
      total_loss=0.0
      correct=0
      total_cnt=0
      cnt_preserve=0
      cnt_change=0
      total_prev_cnt=0
      total_change_cnt=0
                  
      
      for ibatch,(dlatent,attrs,imgname,paired_attrs,imgs) in enumerate(dataloader_gan):
         with torch.no_grad(): 
            sampling=torch.randn((imgs.shape[0],32*(2**2),32,32)).to(device)
            sampling_attr=torch.randn((imgs.shape[0],1,32,32)).to(device)
            
            z,_,_=encoder((dlatent).to(device),paired_attrs.to(device))        
            gen_x_1024=generator(z)
            
            gen_x=F.interpolate(gen_x_1024, size=(256,256))
            imgs_small=F.interpolate(imgs.to(device), size=(256,256))
            _,gen_latents=net(gen_x, randomize_noise=False, resize=opts.resize_outputs, return_latents=True)
            _,ori_latents=net(imgs_small, randomize_noise=False, resize=opts.resize_outputs, return_latents=True)
            
            _,logits=model(gen_latents.view(-1,9216))
            correct+=get_correct_pred_cnt(logits,paired_attrs.to(device),torch.ones_like(logits))
            
            _,ori_logits=model(ori_latents.view(-1,9216))
            ori_pred=get_pred(ori_logits)
            manip_pred=get_pred(logits)
            mask=torch.eq(attrs,paired_attrs).to(device)*torch.eq(ori_pred,attrs.to(device))
            preserve=torch.sum(mask*torch.eq(ori_pred,manip_pred),dim=1)
            cnt_preserve+=preserve.sum()            
            
            t=torch.eq(attrs,paired_attrs).to(device).sum(dim=1)
            mask2=torch.ne(t,torch.ones_like(t)*len(num_classes))
            cnt_change+=(mask2*torch.eq(manip_pred[:,str2catid[cat]],paired_attrs[:,str2catid[cat]].to(device))).sum()
            
            total_cnt+=len(imgs)
            total_prev_cnt+=torch.sum(mask)
            total_change_cnt+=len(imgs)#torch.sum(mask2)
            for ii in range(len(imgs)):
                 compare=np.concatenate((imgs[ii].cpu().numpy(),gen_x_1024[ii].detach().cpu().numpy()),axis=2)
                 save_img_from_torch(compare,cat+'_'+imgname[ii],imgfolder='/projectnb/ivc-ml/nnli/ours_celeb/')
         if ibatch%100==0:
              print('total loss=%.3f at img %d'%(total_loss/(ibatch+1),(ibatch+1)*batch_size))
              print('accu=',correct/total_cnt)
              print('p/c=',(cnt_preserve/total_prev_cnt),(cnt_change/total_change_cnt))              
      print('accu=',100*correct/total_cnt,np.mean(100*correct/total_cnt))
      print('p/c=',(cnt_preserve/total_prev_cnt),(cnt_change/catid2num[str2catid[cat]]),total_change_cnt) #total_change_cnt
      return (cnt_preserve/total_prev_cnt),(cnt_change/catid2num[str2catid[cat]])

            
train_data=CelebaAttrDataset(split='train')
test_data=CelebaAttrDataset(split='test')
val_data=CelebaAttrDataset(split='val')

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 
test_dataloader = DataLoader(test_data, batch_size=10, shuffle=False) 
val_dataloader = DataLoader(val_data, batch_size=10, shuffle=False) 

device = torch.device('cuda')
model = LCNet()
#model.load_state_dict(torch.load('latent_classifier_epoch_20.pth'))
#model.fcs[-1]=nn.Linear(512, sum(num_classes))
model=model.to(device)

params=[par for par in model.parameters()]# if par.requires_grad] 
optimizer=torch.optim.Adam(params,lr=1e-4, betas=(0.5, 0.99))
print(len(params))

test_opts = TestOptions().parse()
# update test options with options used during training
ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
opts = ckpt['opts']
opts.update(vars(test_opts))
if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
if 'output_size' not in opts:
        opts['output_size'] = 1024
opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()

cat2bi={
'hair-color': ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair','Bald'],
'beard': ['Mustache', 'No_Beard', 'Sideburns'],
'hair-type': ['Bald_hair_type', 'Straight_Hair', 'Wavy_Hair'],
'smiling': ['No Smiling','Smiling'],
'eyeglasses': ['No Eyeglasses','Eyeglasses'],
'gender': ['No Male','Male'],
'hat': ['No Wearing_Hat','Wearing_Hat'],
'age': ['Old','Young']
}
str2catid={v:catid for catid,vs in enumerate(cat2bi.values()) for v in vs }
catid2num=[3351,756,1696,986,979,812,936,968]
cats=[]
for vs in cat2bi.values():
    for v in vs:
       cats.append(v)

pretrain=True
if pretrain:
    checkpoint = torch.load('attr_cls/celebaHQ/model_19.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    start_epoch=checkpoint['epoch']+1
    batch_size=checkpoint['batch_size']    
    
    #get_gallery(model)
    #assert 1==0
    
    encoder=Encoder(num_classes=num_classes,device=device)
    encoder.to(device)
    generator=MyGenerator(num_classes=num_classes,device=device)
    generator.to(device)
    enc_params=[par for par in encoder.parameters()] 
    gen_params=[par for par in generator.parameters()]
    optimizerE=torch.optim.Adam(enc_params,lr=1e-3, betas=(0, 0.99))
    optimizerG=torch.optim.Adam(gen_params,lr=1e-3, betas=(0, 0.99))
  
    for ckpt in [6]:
       print('------------ckpt:%d'%ckpt)
       checkpoint = torch.load('celeb/feats_loss/combined/model_%d.pth'%ckpt)
       encoder.load_state_dict(checkpoint['encoder_state_dict'])
       generator.load_state_dict(checkpoint['generator_state_dict'])
       optimizerE.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
       optimizerG.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    
       R_THRESH=[5,20]
       accu={x:[] for x in R_THRESH}
       f=None
       l=None
       attr_accu=[]
       for cat in cats:
         p_rates=[]
         c_rates=[]
         
         #r,f,l=retrieve(device,encoder,generator,model,cat,net,str2catid,catid2num,all_feats=f,all_label=l)
         #for x in R_THRESH:
         # accu[x].append(r[x])
         print('==========================================================================')
         
         for thresh in [101]:#range(10,101,10):
           print('cat:',cat,'thresh:',thresh)
           pr,cr=test_gan(device,encoder,generator,model,cat,net,str2catid,catid2num)
           p_rates.append(pr.cpu().item())
           c_rates.append(cr.cpu().item())        
         print(cat,p_rates,c_rates)
         attr_accu.extend(c_rates)
         print('=============================================')
       print(np.sum(attr_accu)/8)
    print({x:np.sum(accu[x])/8 for x in accu.keys()})# not mean!02/09/2022
    
    assert 1==0
    
num_epochs=20
for e in range(start_epoch,num_epochs):  
      param_dict={'epoch':e, 'batch_size':batch_size,            
               'model_state_dict':model.state_dict(),               
               'optimizer_state_dict':optimizer.state_dict(),
               }
      train(e,device,model,train_dataloader,optimizer,params)
      torch.save(param_dict,'attr_cls/celebaHQ/model_%d.pth'%(e))
      test(device,model,val_dataloader)
      