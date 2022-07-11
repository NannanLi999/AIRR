import torch
import torch.nn as nn
import functools
import numpy as np
import torch.nn.functional as F

class residual_block(nn.Module):
      def __init__(self,dim):
        super(residual_block,self).__init__()        
        self.block= nn.Sequential(nn.ReflectionPad2d(1),#tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
                         nn.Conv2d(dim,dim, kernel_size=3, padding=0, bias=False),#keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
                         nn.BatchNorm2d(dim),#Norm()(h)
                         nn.ReLU(),#tf.nn.relu(h)

                         nn.ReflectionPad2d(1),#tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
                         nn.Conv2d(dim,dim, kernel_size=3, padding=0, bias=False),#keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h) 
                         nn.BatchNorm2d(dim)#Norm()(h)
                        )
      def forward(self,x):
             identity=x
             h=self.block(x)
             h+=identity
             return h
             
class equalized_transform(nn.Module):
      def __init__(self,dim):
        super(equalized_transform,self).__init__()        
        self.block= nn.Sequential(
                         nn.ReflectionPad2d(1),#tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
                         nn.Conv2d(dim,dim, kernel_size=3, padding=0, bias=False),#keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h) 
                         nn.BatchNorm2d(dim),#Norm()(h)
                         nn.ReLU(),#tf.nn.relu(h)

                         nn.ReflectionPad2d(1),#tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
                         nn.Conv2d(dim,dim, kernel_size=3, padding=0, bias=False),#keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h) 
                         nn.BatchNorm2d(dim)
                        )
      def forward(self,x):
             h=self.block(x)
             return h
        
def from_RGB(output_nc):
     h = nn.Sequential(nn.ReflectionPad2d(3),#tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
                       nn.Conv2d(3, output_nc, kernel_size=7, padding=0, bias=False),
                       nn.BatchNorm2d(output_nc),#Norm()(h)
                       nn.ReLU()#tf.nn.relu(h)
                       )
     return h
     
def to_RGB(input_nc):
     h = nn.Sequential(nn.ReflectionPad2d(3),#tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
                       nn.Conv2d(input_nc, 3, kernel_size=7, padding=0, bias=True),
                       nn.Tanh()#tf.nn.tanh(h)
                       )
     return h
     
     
def down_sampling(input_nc, output_nc):     
        h = nn.Sequential(nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=2,padding=1, bias=False),
                          nn.BatchNorm2d(output_nc),#Norm()(h)
                          nn.ReLU()
                          )
        return h
        
def up_sampling(input_nc, output_nc):     
        h = nn.Sequential(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2,bias=False, padding=1, output_padding=1),
                          nn.BatchNorm2d(output_nc),
                          nn.ReLU()
                          )
        return h
        
def dis_from_RGB(output_nc):
     h=nn.Sequential(nn.Conv2d(3, output_nc, kernel_size=4, stride=2, padding=1,bias=True),
                       nn.BatchNorm2d(output_nc),
                       nn.LeakyReLU(0.2)
                       )
     return h
     
def dis_down_sampling(input_nc, output_nc,stride=2):
        if stride==1:
           h = nn.Sequential(nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=stride, padding=1, bias=False),
                          nn.BatchNorm2d(output_nc),#Norm()(h)
                          nn.LeakyReLU(0.2)#tf.nn.relu(h)
                          )
        else:
           h = nn.Sequential(nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=stride, padding=1,bias=False),
                          nn.BatchNorm2d(output_nc),#Norm()(h)
                          nn.LeakyReLU(0.2)#tf.nn.relu(h)
                          )
        return h

        
def build_z(ngf):
     h=[from_RGB(ngf)]
     dim=ngf
     for i in range(2):
        h+=[down_sampling(dim,dim*2)]
        dim=dim*2
     for i in range(9):
        h+= [residual_block(dim)]
     h=nn.Sequential(*h)   
     return h 
     
def build_g(ngf):
     dim=ngf*2**3
     h=[]
     for i in range(2):
        h+=[up_sampling(dim,dim//2)]
        dim=dim//2
     h+=[to_RGB(dim)]
     h=nn.Sequential(*h)   
     return h #Nx3x32x32

class build_d_backbone(nn.Module):
  def __init__(self,ngf):
    super(build_d_backbone,self).__init__()  
    dim=ngf
    h=[dis_from_RGB(dim)]
    for i in range(2):
        h+=[dis_down_sampling(dim,dim*2,stride=2)]
        dim*=2 
    h+=[dis_down_sampling(dim,dim*2,stride=1)]
    self.net=nn.Sequential(*h)
    
  def forward(self,x):   
    return self.net(x)
    
def build_score_map(dim):
    h=nn.Conv2d(dim,1, kernel_size=4, stride=1, padding=1,bias=True)
    return h

class build_classifier(nn.Module):
  def __init__(self,dim,num_classes):
    super(build_classifier,self).__init__()
    
    h=[residual_block(dim)]
    h+=[down_sampling(dim,2*dim)]   
    h+=[nn.AvgPool2d(8)]
    h+=[nn.Conv2d(2*dim,num_classes, kernel_size=1,bias=True)]
    self.net=nn.Sequential(*h)  
  
  def forward(self,x):  
    x=self.net(x).squeeze(3).squeeze(2)
    return x
  
class build_encoder_classifier(nn.Module):
  def __init__(self,dim,num_classes):
    super(build_encoder_classifier,self).__init__()
    
    h=[down_sampling(dim,2*dim)]
    h+=[nn.AvgPool2d(16)]
    h+=[nn.Conv2d(2*dim,num_classes, kernel_size=1,bias=True)]
    self.net=nn.Sequential(*h)  
    
  def forward(self,x):   
    x=self.net(x).squeeze(3).squeeze(2)
    return x
       
class Encoder(nn.Module):
    def __init__(self,ngf=16,num_classes=[17,4]):
          super(Encoder,self).__init__()
          
          self.ngf=ngf
          self.networkZ=build_z(ngf)
          self.networkZ_body=build_z(ngf)
          

          self.embeds=nn.Embedding(sum(num_classes),64,max_norm=100)
          self.num_classes=num_classes
          
          self.classifier=build_encoder_classifier(ngf*4,sum(num_classes))
          self.filter=equalized_transform(ngf*4)
          
          self.fc=nn.Linear(64,ngf*4*2)
        
          
    def forward(self,imgs,attrs,mask=None):
    

          z_body=self.networkZ_body((1-mask)*imgs)
          z_before=self.networkZ(mask*imgs) 
    
          attr_logits_real=self.classifier(z_before)
                    
          z=self.filter(z_before)
          
          attr_logits_fake=self.classifier(z)

          cur_index=0
          attr_feats=0
          for i in range(len(self.num_classes)):
            attr_feats+=self.embeds(cur_index+attrs[:,i])  
            cur_index+=self.num_classes[i]
            
          attr_feats=self.fc(attr_feats).view(-1,self.ngf*4*2,1,1)       
          mean,std=torch.chunk(attr_feats, chunks=2, dim=1)        
          z=(1+std)*z+mean
          
          z=torch.cat((z,z_body),dim=1)
   
          return z,attr_logits_real,attr_logits_fake
          
class Generator(nn.Module):
    def __init__(self,ngf=16):
          super(Generator,self).__init__()              
         
          self.networkG= build_g(ngf)
          
    def forward(self,z):
         
         nimgs=self.networkG(z)
         return nimgs
          
class Discriminator(nn.Module):
    def __init__(self,ngf=16,num_classes=[17,4]):
          super(Discriminator,self).__init__()          
        
          self.networkD_backbone=build_d_backbone(ngf)
          self.build_score_map=build_score_map(ngf*8)
          self.networkD_classifier=build_classifier(ngf*8,sum(num_classes))
          
    def forward(self,imgs):
        backbone=self.networkD_backbone(imgs)
        score_map=self.build_score_map(backbone)
        
        logits=self.networkD_classifier(backbone)       
        return score_map,logits
        
