import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from dataloader import DeepfashionAttrDataset,DeepfashionDataset
from model7_lh import *
from PIL import Image
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision
import pickle

batch_size=16
start_epoch=0

MSELoss=torch.nn.MSELoss(reduction='mean')
SCELoss=nn.CrossEntropyLoss(reduction='mean')
num_classes=[17,4]

def save_img_from_torch(img,imgname,img_mean,imgfolder='output/'):
   img=np.clip((img+1)/2.0,0,1)#np.clip(img+img_mean,0,1)#
   img=np.transpose(img,[1,2,0])
   name="_".join(imgname.split('/')[-2:]).split('.')[0]+'.png'
   nimg=Image.fromarray(np.uint8(img*255))
   nimg.save(imgfolder+name)
        
def get_attr_loss(logits,labels):
       cur_index=0
       attr_loss=0
       for i in range(len(num_classes)):
           attr_loss+=SCELoss(logits[:,cur_index:cur_index+num_classes[i]],labels[:,i])#           
           cur_index+=num_classes[i]
       return attr_loss
       
def get_correct_pred_cnt(logits,labels):    
     cur_index=0
     correct=[]
     for i in range(len(num_classes)):
         pred=torch.argmax(logits[:,cur_index:cur_index+num_classes[i]],dim=1)#
         correct.append(torch.eq(pred,labels[:,i]).to(dtype=torch.float32).sum().cpu().item())
         cur_index+=num_classes[i]
     return np.array(correct)

def train(epoch,device,model,dataloader,optimizer,params):
     print("=======adv_train=========") 
     model.train()
     total_loss=0
     for ibatch,(imgs,attrs) in enumerate(dataloader): 
           if epoch>0:
               for param in params:
                     param.requires_grad = True 
           optimizer.zero_grad()  
           logits=model(imgs.to(device))
           loss=get_attr_loss(logits,attrs.to(device))
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
      for ibatch,(imgs,attrs) in enumerate(dataloader):
         with torch.no_grad(): 
           logits=model(imgs.to(device))
           loss=get_attr_loss(logits,attrs.to(device))
           total_loss+=loss.cpu().item()
           correct+=get_correct_pred_cnt(logits,attrs.to(device))
           total_cnt+=len(imgs)
         if ibatch%100==0:
              print('total loss=%.3f at img %d'%(total_loss/(ibatch+1),(ibatch+1)*batch_size))
              print('accu=',correct/total_cnt)
      print('accu=',100*correct/total_cnt,np.mean(100*correct/total_cnt))
      
def test_gan(device,encoder,generator,model,cat,thresh):     
      print("=======test-gan=========")  
      data_gan=DeepfashionDataset(split='test',get_paired_seg=True,thresh=thresh,cat=cat) 
      dataloader_gan = DataLoader(data_gan, batch_size=10, shuffle=False)
      model.eval()  
      encoder.eval()
      generator.eval()    
      total_loss=0.0
      correct=0
      total_cnt=0
      for ibatch,(imgs,attrs,_,paired_attrs,seg,_,paired_seg) in enumerate(dataloader_gan):
         with torch.no_grad(): 
            sampling=torch.randn((imgs.shape[0],32*(2**2),32,32)).to(device)
            sampling_attr=torch.randn((imgs.shape[0],1,32,32)).to(device)
            paired_seg=paired_seg.to(device)
            small_seg=F.interpolate(paired_seg,size=14,mode='bilinear')#26#4
            small_seg=torch.gt(small_seg,0).float()
            z,_,_,_,_=encoder(imgs.to(device),sampling,sampling_attr,paired_attrs.to(device),cat=paired_seg)#small_seg       
            gen_x=generator(z)
            face_seg=seg.to(device)
            gen_x=gen_x*(1-face_seg)+imgs.to(device)*face_seg
            
            logits=model(gen_x.to(device))
            correct+=get_correct_pred_cnt(logits,paired_attrs.to(device))
            total_cnt+=len(imgs)
         if ibatch%100==0:
              print('total loss=%.3f at img %d'%(total_loss/(ibatch+1),(ibatch+1)*batch_size))
              print('accu=',correct/total_cnt)
      print('accu=',100*correct/total_cnt,np.mean(100*correct/total_cnt))
      
def test_amgan(device,encoder,generator,model,thresh):     
      print("=======test-gan=========")  
      data_gan=DeepfashionDataset(split='test',get_ref_img=True,thresh=thresh) 
      dataloader_gan = DataLoader(data_gan, batch_size=10, shuffle=False) 
      model.eval()  
      encoder.eval()
      generator.eval()    
      total_loss=0.0
      correct=0
      total_cnt=0
      for ibatch,(imgs,attrs,_,paired_attrs,_,_,attrs_onehot,paired_attrs_onehot) in enumerate(dataloader_gan):
         with torch.no_grad(): 
            sampling=torch.randn((imgs.shape[0],32*(2**2),32,32)).to(device)
            sampling_attr=torch.randn((imgs.shape[0],1,32,32)).to(device)
            z,_=encoder(imgs.to(device),paired_attrs_onehot.to(device))           
            gen_x,att=generator(z)
            gen_x=gen_x*att+imgs.to(device)*(1-att)
            logits=model(gen_x.to(device))
            correct+=get_correct_pred_cnt(logits,paired_attrs.to(device))
            total_cnt+=len(imgs)
         if ibatch%100==0:
              print(paired_attrs[0],attrs[0],paired_attrs_onehot[0],attrs_onehot[0])
              print('total loss=%.3f at img %d'%(total_loss/(ibatch+1),(ibatch+1)*batch_size))
              print('accu=',correct/total_cnt)
      print('accu=',100*correct/total_cnt,np.mean(100*correct/total_cnt))

def test_stargan(device,encoder,generator,model,thresh):     
      print("=======test-gan=========")  
      data_gan=DeepfashionDataset(split='test',get_ref_img=True,thresh=thresh) 
      dataloader_gan = DataLoader(data_gan, batch_size=10, shuffle=False)
      model.eval()  
      encoder.eval()
      generator.eval()    
      total_loss=0.0
      correct=0
      total_cnt=0
      for ibatch,(imgs,attrs,_,paired_attrs,_,_,attrs_onehot,paired_attrs_onehot) in enumerate(dataloader_gan):
         with torch.no_grad(): 
            z,_=encoder(imgs.to(device),paired_attrs_onehot.to(device))           
            gen_x=generator(z)
            logits=model(gen_x.to(device))
            correct+=get_correct_pred_cnt(logits,paired_attrs.to(device))
            total_cnt+=len(imgs)
         if ibatch%100==0:
              #print(paired_attrs[0],attrs[0],paired_attrs_onehot[0],attrs_onehot[0])
              print('total loss=%.3f at img %d'%(total_loss/(ibatch+1),(ibatch+1)*batch_size))
              print('accu=',correct/total_cnt)
      print('accu=',100*correct/total_cnt,np.mean(100*correct/total_cnt))
      
def getfeatures_amgan(device,encoder,generator,cnn,dataobject,gan_dataloader):     
      print("=======test-gan=========")  
      model.eval()  
      encoder.eval()
      generator.eval()    
      total_loss=0.0
      correct=0
      total_cnt=0
      for ibatch,(imgs,attrs,names,paired_attrs,_,_,attrs_onehot,paired_attrs_onehot) in enumerate(gan_dataloader):
         with torch.no_grad(): 
            sampling=torch.randn((imgs.shape[0],32*(2**2),32,32)).to(device)
            sampling_attr=torch.randn((imgs.shape[0],1,32,32)).to(device)
            z,_=encoder(imgs.to(device),paired_attrs_onehot.to(device))           
            gen_x,att=generator(z)
            gen_x=gen_x*att+imgs.to(device)*(1-att)
            feats=cnn(gen_x.to(device)).view(-1,2048)#N,2048,1,1
            feats=feats.cpu().numpy()
            newa=paired_attrs.numpy()
            for i in range(len(names)):            
                 label='@'+str(newa[i,0])+'_'+str(newa[i,1])+'@'
                 pickle.dump(feats[i],open('resnet_features/amgan_sleeve/'+"_".join(names[i].split('/')[-2:]).split('.')[0]+label+'.pkl','wb'))
         if ibatch%100==0:
            print(ibatch)
            
def getfeatures_gan(device,encoder,generator,cnn,dataobject,gan_dataloader):     
      print("=======test-gan=========")  
      model.eval()  
      encoder.eval()
      generator.eval()    
      total_loss=0.0
      correct=0
      total_cnt=0
      for ibatch,(imgs,attrs,names,paired_attrs,_,_,paired_seg) in enumerate(gan_dataloader):
         with torch.no_grad(): 
            sampling=torch.randn((imgs.shape[0],32*(2**2),32,32)).to(device)
            sampling_attr=0
            z,_=encoder(imgs.to(device),sampling,sampling_attr,paired_attrs.to(device))           
            gen_x=generator(z)
            paired_seg=paired_seg.to(device)
            gen_x=gen_x*paired_seg+imgs.to(device)*(1-paired_seg)
            feats=cnn(gen_x.to(device)).view(-1,2048)#N,2048,1,1
            feats=feats.cpu().numpy()
            newa=paired_attrs.numpy()
            for i in range(len(names)):            
                 label='@'+str(newa[i,0])+'_'+str(newa[i,1])+'@'
                 pickle.dump(feats[i],open('resnet_features/gan_sleeve/'+"_".join(names[i].split('/')[-2:]).split('.')[0]+label+'.pkl','wb'))
         if ibatch%100==0:
            print(ibatch)
                     
def getfeatures_resnet(device,cnn,dataloader):     
      print("=======test=========")  
      model.eval()      
      total_loss=0.0
      correct=np.zeros(len(num_classes))
      total_cnt=0
      for ibatch,(imgs,attrs,names) in enumerate(dataloader):
         with torch.no_grad(): 
           feats=cnn(imgs.to(device)).view(-1,2048)
           feats=feats.cpu().numpy()
           a=attrs.numpy()
         for i in range(len(names)):            
                 label='@'+str(a[i,0])+'_'+str(a[i,1])+'@'
                 pickle.dump(feats[i],open('resnet_features/ori/'+"_".join(names[i].split('/')[-2:]).split('.')[0]+label+'.pkl','wb'))
         if ibatch%100==0:
              print(ibatch)
            
train_data=DeepfashionAttrDataset(split='train')
test_data=DeepfashionAttrDataset(split='test')
val_data=DeepfashionAttrDataset(split='val')

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 
test_dataloader = DataLoader(test_data, batch_size=10, shuffle=False) 
val_dataloader = DataLoader(val_data, batch_size=10, shuffle=False) 

device = torch.device('cuda')
model = torchvision.models.resnet50(pretrained=True)#torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
for param in model.parameters():
    param.requires_grad = False   
model.fc=nn.Linear(2048, sum(num_classes))
model=model.to(device)
cnn = nn.Sequential(*list(model.children())[:-1])

params=[par for par in model.parameters()]# if par.requires_grad] 
optimizer=torch.optim.Adam(params,lr=1e-5, betas=(0.5, 0.99))
print(len(params))

pretrain=True
if pretrain:
    checkpoint = torch.load('attr_cls/1e-4/model_25.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    start_epoch=checkpoint['epoch']+1
    batch_size=checkpoint['batch_size']    
    #test(device,model,test_dataloader)

    encoder=Encoder(ngf=32, num_classes=[17,4],device=device)
    encoder.to(device)
    generator=Generator(ngf=32, num_classes=[17,4],device=device)
    generator.to(device)
    enc_params=[par for par in encoder.parameters()] 
    gen_params=[par for par in generator.parameters()]
    optimizerE=torch.optim.Adam(enc_params,lr=1e-3, betas=(0, 0.99))
    optimizerG=torch.optim.Adam(gen_params,lr=1e-3, betas=(0, 0.99))
  
    checkpoint = torch.load('info/hloss/model_30.pth')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    optimizerE.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    optimizerG.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    for cat in range(2):
        print('cat:',cat)  
        test_gan(device,encoder,generator,model,cat,101)
        #test_amgan(device,encoder,generator,model,thresh)
        #test_stargan(device,encoder,generator,model,thresh)
        print('==========================================================================')
    #getfeatures_amgan(device,encoder,generator,cnn,test_data,dataloader_gan)
    #getfeatures_resnet(device,cnn,test_dataloader)
    #getfeatures_gan(device,encoder,generator,cnn,test_data,dataloader_gan)
    assert 1==0
num_epochs=30
for e in range(start_epoch,num_epochs):  
      param_dict={'epoch':e, 'batch_size':batch_size,            
               'model_state_dict':model.state_dict(),               
               'optimizer_state_dict':optimizer.state_dict(),
               }
      train(e,device,model,train_dataloader,optimizer,params)
      torch.save(param_dict,'attr_cls/1103/model_%d.pth'%(e))
      test(device,model,val_dataloader)
      