import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from dataloader_celebm import CelebaAttrDataset,CelebaDataset
from model7_celebm import *
from PIL import Image
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision
import pickle
import os
import argparse

batch_size=16
start_epoch=0

SCELoss=nn.CrossEntropyLoss(reduction='none')
num_classes=[5,3,3,2,2,2,2,2]


gallery_dir='resnet_features/celeb/gallery/'
R_THRESH=[5,20]
def get_target_matrix():
     tgt_dir=gallery_dir
     tgt_names=sorted([x for x in os.listdir(tgt_dir)])
     tgt_matrix=np.zeros((len(tgt_names),2048))
     index2name={}
     for i,name in enumerate(tgt_names):
         feats=pickle.load(open(tgt_dir+name,'rb'))
         tgt_matrix[i]=feats
         index2name[i]=name.replace('.pkl','.jpg')
     return tgt_matrix,index2name
     
def retrieve(device,model,encoder,generator,cat,tgt_matrix=None,index2name=None):    
     data_gan=CelebaDataset(split='test',get_paired_seg=True,cat=cat)
     dataloader_gan = DataLoader(data_gan, batch_size=10, shuffle=False)
     
     model.eval()
     encoder.eval()
     generator.eval() 
     
     if tgt_matrix is None:
         tgt_matrix,index2name=get_target_matrix()
     tnorm=np.linalg.norm(tgt_matrix,ord=2,axis=1)
     hit={x:0 for x in R_THRESH}
     for ibatch,(imgs,attrs,name,paired_attrs,paired_imgs,mask) in enumerate(dataloader_gan):
           with torch.no_grad(): 
                    
              z,_,_,_,attrs_feats=encoder((imgs).to(device),1,1,paired_attrs.to(device),cat=mask.to(device))           
              gen_x=generator(z,attrs_feats)             
              gen_feats=cnn(gen_x).view(-1,2048).cpu().numpy()
              paired_attrs=paired_attrs.cpu().numpy()
              
           for i in range(len(gen_feats)):
               feats=gen_feats[i]
               sim=np.sum(feats[None,:]*tgt_matrix,axis=1)/(np.linalg.norm(feats,ord=2)*tnorm)
               sim=np.abs(sim)
               sorted_sim=np.argsort(-sim)
               for th in R_THRESH:
                  indexes=sorted_sim[:th]
                  rnames=[index2name[x] for x in indexes]
                  cand_attrs=np.array([data_gan.attr_dict[n] for n in rnames])#.replace('/','-')
                  t=np.equal(cand_attrs,paired_attrs[i])
                  if len(num_classes) in np.sum(t,axis=1):
                    hit[th]+=1
                    #print(name[i],np.array(rnames)[np.equal(np.sum(t,axis=1),len(num_classes))])
                    #print(cat,attrs[i],paired_attrs[i])
                   
           #if ibatch%10==0:
           #     print(ibatch,{h:hit[h]/((ibatch+1)*10) for h in hit.keys()})       
     r={h:hit[h]/len(data_gan.index) for h in hit.keys()}      
     return r,tgt_matrix,index2name

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
     if epoch>0:
        for param in params:
            param.requires_grad = True 
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
      pickle.dump(pseudo_attr,open('/net/ivcfs4/mnt/data/nnli/Celeba/name2attr.pkl','wb'))
      
def test_stargan(device,encoder,generator,model,cat,thresh):     
      print("=======test-gan=========")  
      data_gan=CelebaDataset(split='test',get_ref_img=True,cat=cat,thresh=thresh) 
      dataloader_gan = DataLoader(data_gan, batch_size=10, shuffle=False)
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
      for ibatch,(imgs,attrs,_,paired_attrs,_,_,attrs_onehot,paired_attrs_onehot) in enumerate(dataloader_gan):
         with torch.no_grad(): 
            sampling=torch.randn((imgs.shape[0],32*(2**2),32,32)).to(device)
            sampling_attr=torch.randn((imgs.shape[0],1,32,32)).to(device)
            z,_=encoder(imgs.to(device),paired_attrs_onehot.to(device))           
            gen_x=generator(z)           
            logits=model(gen_x.to(device))
            correct+=get_correct_pred_cnt(logits,paired_attrs.to(device),torch.ones_like(logits))
            
            ori_logits=model(imgs.to(device))
            ori_pred=get_pred(ori_logits)
            manip_pred=get_pred(logits)
            mask=torch.eq(attrs,paired_attrs).to(device)*torch.eq(ori_pred,attrs.to(device))
            preserve=torch.sum(mask*torch.eq(ori_pred,manip_pred),dim=1)
            cnt_preserve+=preserve.sum()            
            
            t=torch.eq(attrs,paired_attrs).to(device).sum(dim=1)
            mask2=torch.ne(t,torch.ones_like(t)*len(num_classes))
            cnt_change+=(mask2*torch.eq(manip_pred[:,cat],paired_attrs[:,cat].to(device))).sum()
            
            total_cnt+=len(imgs)
            total_prev_cnt+=torch.sum(mask)
            total_change_cnt+=len(imgs)#torch.sum(mask2)
         if ibatch%100==0:
              print('total loss=%.3f at img %d'%(total_loss/(ibatch+1),(ibatch+1)*batch_size))
              print('accu=',correct/total_cnt)
              print('p/c=',(cnt_preserve/total_prev_cnt),(cnt_change/total_change_cnt))
      print('accu=',100*correct/total_cnt,np.mean(100*correct/total_cnt))
      print('p/c=',(cnt_preserve/total_prev_cnt),(cnt_change/total_change_cnt),total_change_cnt)
      return (cnt_preserve/total_prev_cnt),(cnt_change/total_change_cnt)
      
      
def test_gan(device,encoder,generator,model,cat,thresh):     
      #print("=======test-gan=========")  
      data_gan=CelebaDataset(split='test',get_paired_seg=True,cat=cat,thresh=thresh) 
      dataloader_gan = DataLoader(data_gan, batch_size=10, shuffle=False)
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
      mse=0
      for ibatch,(imgs,attrs,name,paired_attrs,paired_imgs,mask) in enumerate(dataloader_gan):
         with torch.no_grad(): 
            sampling=torch.randn((imgs.shape[0],32*(2**2),32,32)).to(device)
            sampling_attr=torch.randn((imgs.shape[0],1,32,32)).to(device)
            
            z,_,_,_,_=encoder(imgs.to(device),sampling,sampling_attr,paired_attrs.to(device),cat=mask.to(device))#small_seg       
            gen_x=generator(z,1)
            
            z_rec,_,_,_,_=encoder(imgs.to(device),sampling,sampling_attr,attrs.to(device),cat=mask.to(device))#small_seg       
            gen_rec=generator(z_rec,1)
            
            logits=model(gen_x.to(device))
            correct+=get_correct_pred_cnt(logits,paired_attrs.to(device),torch.ones_like(logits))
            
            ori_logits=model(imgs.to(device))
            ori_pred=get_pred(ori_logits)
            manip_pred=get_pred(logits)
            mask=torch.eq(attrs,paired_attrs).to(device)*torch.eq(ori_pred,attrs.to(device))
            preserve=torch.sum(mask*torch.eq(ori_pred,manip_pred),dim=1)
            cnt_preserve+=preserve.sum()                    
            
            t=torch.eq(attrs,paired_attrs).to(device).sum(dim=1)
            mask2=torch.ne(t,torch.ones_like(t)*len(num_classes))
            cnt_change+=(mask2*torch.eq(manip_pred[:,cat],paired_attrs[:,cat].to(device))).sum()
            
            total_cnt+=len(imgs)
            total_prev_cnt+=torch.sum(mask)
            total_change_cnt+=len(imgs)#torch.sum(mask2)
            mse+=torch.mean(torch.square(gen_rec-imgs.to(device)),dim=(1,2,3)).sum()
            """if ibatch%100==0:
              print('total loss=%.3f at img %d'%(total_loss/(ibatch+1),(ibatch+1)*batch_size))
              print('accu=',correct/total_cnt)
              print('p/c=',(cnt_preserve/total_prev_cnt),(cnt_change/total_change_cnt))"""
      #print('accu=',100*correct/total_cnt,np.mean(100*correct/total_cnt))
      #print('p/c=',(cnt_preserve/total_prev_cnt),(cnt_change/total_change_cnt),total_change_cnt)
      return (cnt_preserve/total_prev_cnt),(cnt_change/total_change_cnt),mse/total_cnt
      
def test_amgan(device,encoder,generator,model,cat,thresh):     
      print("=======test-gan=========")  
      data_gan=CelebaDataset(split='test',get_ref_img=True,cat=cat,thresh=thresh) 
      dataloader_gan = DataLoader(data_gan, batch_size=10, shuffle=False)
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
      for ibatch,(imgs,attrs,_,paired_attrs,_,_,attrs_onehot,paired_attrs_onehot) in enumerate(dataloader_gan):
         with torch.no_grad(): 
            sampling=torch.randn((imgs.shape[0],32*(2**2),32,32)).to(device)
            sampling_attr=torch.randn((imgs.shape[0],1,32,32)).to(device)
            z,_=encoder(imgs.to(device),paired_attrs_onehot.to(device))           
            gen_x,att=generator(z)
            gen_x=gen_x*att+imgs.to(device)*(1-att)
            logits=model(gen_x.to(device))
            correct+=get_correct_pred_cnt(logits,paired_attrs.to(device),torch.ones_like(logits))
            
            ori_logits=model(imgs.to(device))
            ori_pred=get_pred(ori_logits)
            manip_pred=get_pred(logits)
            mask=torch.eq(attrs,paired_attrs).to(device)*torch.eq(ori_pred,attrs.to(device))
            preserve=torch.sum(mask*torch.eq(ori_pred,manip_pred),dim=1)
            cnt_preserve+=preserve.sum()            
            
            t=torch.eq(attrs,paired_attrs).to(device).sum(dim=1)
            mask2=torch.ne(t,torch.ones_like(t)*len(num_classes))
            cnt_change+=(mask2*torch.eq(manip_pred[:,cat],paired_attrs[:,cat].to(device))).sum()
            
            total_cnt+=len(imgs)
            total_prev_cnt+=torch.sum(mask)
            total_change_cnt+=len(imgs)#torch.sum(mask2)
         if ibatch%100==0:
              print('total loss=%.3f at img %d'%(total_loss/(ibatch+1),(ibatch+1)*batch_size))
              print('accu=',correct/total_cnt)
              print('p/c=',(cnt_preserve/total_prev_cnt),(cnt_change/total_change_cnt))
      print('accu=',100*correct/total_cnt,np.mean(100*correct/total_cnt))
      print('p/c=',(cnt_preserve/total_prev_cnt),(cnt_change/total_change_cnt),total_change_cnt)
      return (cnt_preserve/total_prev_cnt),(cnt_change/total_change_cnt)

"""            
train_data=CelebaAttrDataset(split='train')
test_data=CelebaAttrDataset(split='test')
val_data=CelebaAttrDataset(split='val')

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 
test_dataloader = DataLoader(test_data, batch_size=10, shuffle=False) 
val_dataloader = DataLoader(val_data, batch_size=10, shuffle=False) 
"""
device = torch.device('cuda')
model = torchvision.models.resnet50(pretrained=True)#torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
for param in model.parameters():
    param.requires_grad = False   
model.fc=nn.Linear(2048, sum(num_classes))
model=model.to(device)
cnn = nn.Sequential(*list(model.children())[:-1])

params=[par for par in model.parameters()]# if par.requires_grad] 
optimizer=torch.optim.Adam(params,lr=1e-4, betas=(0.5, 0.99))
print(len(params))

all_accu={
'0.125_0.5_1.0_1.0': [0.8583125323057175,0.008622161578387022,0.8984375, 0.9638125],
'0.25_0.5_1.0_1.0':[0.8543125465512276,0.00918811303563416,0.8894375, 0.9603125],
'2.0_0.5_1.0_1.0': [0.8755000457167625,0.016578462324105203,0.9043125000000001, 0.964],
'1.0_0.5_1.0_1.0':[0.8591875359416008,0.011444821138866246,0.891875, 0.9583125],

'0.5_0.125_1.0_1.0': [0.8125625401735306,0.022412331542000175,0.848125, 0.9296875],
'0.5_1.0_1.0_1.0': [0.9121875315904617,0.015371710294857621,0.9315625000000001, 0.9731875000000001],
'0.5_2.0_1.0_1.0':[ 0.9281875565648079, 0.020526648499071598,0.93425, 0.977375],
'0.5_0.25_1.0_1.0': [0.848187543451786 0.011513521312735975,0.886125, 0.9545625],

'0.5_0.5_0.25_1.0': [0.9304375424981117,0.02702602930366993,0.9428125, 0.977],
'0.5_0.5_2.0_1.0': [0.8252500370144844,0.007588331704027951,0.8763749999999999, 0.9519375],
'0.5_0.5_0.125_1.0': [0.9526875540614128,0.04497187677770853,0.950125, 0.9815625],
'0.5_0.5_0.5_1.0': [0.8932500407099724,0.01486077718436718,0.922, 0.9705625],

'0.5_0.5_1.0_0.0': [0.5097500216215849,0.008753608562983572,0.6068749999999999, 0.7535],
'0.5_0.5_1.0_0.125': [0.6296875290572643,0.008775848196819425,0.72075, 0.84775],
'0.5_0.5_1.0_2.0': [0.9410000443458557,0.025574556784704328,0.94875, 0.9805625],
'0.5_0.5_1.0_0.25': [0.759625032544136,0.008728203130885959,0.81875, 0.9205625],
'0.5_0.5_1.0_0.5': [0.8456250429153442,0.011593239498324692,0.882375, 0.9515],

'0.5_0.5_1.0_1.0': [0.918500043451786,0.042752842884510756,0.9306875, 0.9748125000000001]
}

pretrain=True
if pretrain:
    checkpoint = torch.load('attr_cls/celeba/model_15.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    start_epoch=checkpoint['epoch']+1
    batch_size=checkpoint['batch_size']    
       
    encoder=Encoder(ngf=32, num_classes=num_classes,device=device)
    encoder.to(device)
    generator=Generator(ngf=32, num_classes=num_classes,device=device)
    generator.to(device)
    enc_params=[par for par in encoder.parameters()] 
    gen_params=[par for par in generator.parameters()]
    optimizerE=torch.optim.Adam(enc_params,lr=1e-3, betas=(0, 0.99))
    optimizerG=torch.optim.Adam(gen_params,lr=1e-3, betas=(0, 0.99))
  
    f=None
    l=None
    for cdir in os.listdir('/projectnb/ivc-ml/nnli/celeb/search/'):
          if cdir in all_accu.keys():
            continue
          checkpoint = torch.load('/projectnb/ivc-ml/nnli/celeb/search/'+cdir+'/model_15.pth')
          encoder.load_state_dict(checkpoint['encoder_state_dict'])
          generator.load_state_dict(checkpoint['generator_state_dict'])
          optimizerE.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
          optimizerG.load_state_dict(checkpoint['generator_optimizer_state_dict'])
          p_rates=[]
          c_rates=[]
          mse_all=[]
          raccu={x:[] for x in R_THRESH}
          for cat in range(8):              
              #print('cat:',cat,'thresh:',thresh)
              pr,cr,mse=test_gan(device,encoder,generator,model,cat,1000)
              #pr,cr=test_amgan(device,encoder,generator,model,cat,thresh)    
              #pr,cr=test_stargan(device,encoder,generator,model,cat,thresh)
              p_rates.append(pr.cpu().item())
              c_rates.append(cr.cpu().item())  
              mse_all.append(mse.cpu().item())      
              #print(cat,p_rates,c_rates)
              
              r,f,l=retrieve(device,model,encoder,generator,cat,tgt_matrix=f,index2name=l)
              for x in R_THRESH:
                   raccu[x].append(r[x])
          print(cdir,np.mean(c_rates),np.mean(mse_all),[np.mean(raccu[x]) for x in raccu.keys()])
          all_accu[cdir]=[np.mean(c_rates),np.mean(mse_all),np.mean(raccu[R_THRESH[0]]),np.mean(raccu[R_THRESH[1]])]
    pickle.dump(all_accu,open('celeb/celeb_abl.pkl','wb'))
    assert 1==0
    
num_epochs=20
for e in range(start_epoch,num_epochs):  
      param_dict={'epoch':e, 'batch_size':batch_size,            
               'model_state_dict':model.state_dict(),               
               'optimizer_state_dict':optimizer.state_dict(),
               }
      train(e,device,model,train_dataloader,optimizer,params)
      torch.save(param_dict,'attr_cls/celeba2/model_%d.pth'%(e))
      test(device,model,val_dataloader)
      