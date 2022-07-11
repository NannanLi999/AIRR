import torch
import torch.nn as nn
import functools
import numpy as np
import torch.nn.functional as F
import math

def get_weight(weight, gain=1, use_wscale=True, lrmul=1):
    fan_in = np.prod(weight.size()[1:]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init
    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        runtime_coef = he_std * lrmul
    else:
        runtime_coef = lrmul
    return weight * runtime_coef
    
class Dense_layer(nn.Module):
    def __init__(self, input_size, output_size, gain=1, use_wscale=True, lrmul=1):
        super(Dense_layer, self).__init__()  
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        self.gain = gain
        self.use_wscale = use_wscale
        self.lrmul = lrmul
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        w = get_weight(self.weight, gain=self.gain, use_wscale=self.use_wscale, lrmul=self.lrmul)
        b = self.bias
        x = F.linear(x, w, bias=b)
        return x
    
def apply_bias_act(x, act='linear', alpha=None, gain=None):
    if act == 'linear':
        return x
    elif act == 'lrelu':
        if alpha is None:
            alpha = 0.2
        if gain is None:
            gain = np.sqrt(2)
        x = F.leaky_relu(x, negative_slope=alpha)
        x = x*gain
        return x
                    
class equalized_transform(nn.Module):
      def __init__(self,dim,shortcut=False):
        super(equalized_transform,self).__init__()        
        self.shortcut=shortcut
        self.dense = nn.ModuleList()
        #nn.Conv1d(dim, dim, 3, padding='same')
        for layer_idx in range(18):
            self.dense.append(Dense_layer(dim,dim))#nn.Linear(dim, dim))
            
      def forward(self,inputs):
             x = inputs.transpose(0,1)#18,N,512
             out = []
             for layer_idx in range(18):
                  out.append(apply_bias_act(self.dense[layer_idx](x[layer_idx]), act='linear'))
             x = torch.stack(out, dim=1)
             if self.shortcut:
               x=x+inputs
             return x
             
channel_multiplier=2
channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
import sys 
sys.path.append('pixel2style2pixel/')

from pixel2style2pixel.models.stylegan2.model import ConvLayer,ResBlock,EqualLinear,Generator
from pixel2style2pixel.models.psp import get_keys
    
class build_d_backbone(nn.Module):
  def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu')
        )
    
  def forward(self,input):   
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)
        return out
    
def build_score_map():
    h=EqualLinear(channels[4], 1)
    return h

class build_classifier(nn.Module):
  def __init__(self):
    super(build_classifier,self).__init__()
    
    self.net=nn.Sequential(
            EqualLinear(channels[4], channels[4]//2, activation='fused_lrelu'),
            EqualLinear(channels[4]//2, 21),
        )
  
  def forward(self,x,mask=None):  
    x=self.net(x)
    return x
  
class build_encoder_classifier(nn.Module):
  def __init__(self):
    super(build_encoder_classifier,self).__init__()
    
    self.net=nn.Sequential(
            EqualLinear(channels[4], channels[4]//2, activation='fused_lrelu'),
            EqualLinear(channels[4]//2, 21),
        )
    
  def forward(self,x):   
    x=self.net(x.mean(dim=1))
    return x

class Encoder(nn.Module):
    def __init__(self,ngf=16,num_classes=[17,4]):
          super(Encoder,self).__init__() 
          
          self.num_classes=num_classes
          self.filter=equalized_transform(channels[4])
          self.classifier=build_encoder_classifier()
          self.embeds=nn.Embedding(sum(num_classes),256,max_norm=200)
          self.fc=Dense_layer(256,512*2)
          
    def forward(self,inputs,attrs,mask=None):
          attr_real_logits=self.classifier(inputs)
          z=self.filter(inputs)
          attr_fake_logits=self.classifier(z)
          
          cur_index=0
          attr_feats=[]
          dic=self.embeds.weight.clone()
          for i in range(len(self.num_classes)):
            index=cur_index+attrs[:,i] 
            
            attr_i=dic[index]#F.linear(dic[index],self.fc.weight)+self.fc.bias/8.0
            """if i==3:
               attr_i_wo=F.linear(dic[index-1],self.fc.weight)+self.fc.bias/8.0
               attr_i=attr_i*0+attr_i_wo*1.0"""
            attr_feats.append(attr_i)
                         
            cur_index=cur_index+self.num_classes[i]
            
          attr_feats=torch.stack(attr_feats,dim=0).sum(dim=0)   
          attr_feats=self.fc(attr_feats).view(-1,1,512*2)#attr_feats.view(-1,1,512*2) #       
          mean,std=torch.chunk(attr_feats, chunks=2, dim=2)        
          z=(1+std)*z+mean
        
          return z,attr_real_logits,attr_fake_logits

class MyGenerator(nn.Module):
    def __init__(self,ngf=16):
          super(MyGenerator,self).__init__()              
         
          self.networkG= Generator(1024, 512, 8)         
          
    def forward(self,z):
         
         nimgs, _ = self.networkG([z], input_is_latent=True, randomize_noise=False)
         return nimgs
         
    def init_generator(self):
          state_dict = torch.load('pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', map_location='cpu')
          self.networkG.load_state_dict(get_keys(state_dict, 'decoder'), strict=True)  
              
class Discriminator(nn.Module):
    def __init__(self,ngf=16,num_classes=[17,4]):
          super(Discriminator,self).__init__()          
        
          self.networkD_backbone=build_d_backbone(1024)
          self.build_score_map=build_score_map()
          self.networkD_classifier=build_classifier()
          
    def forward(self,imgs):
        backbone=self.networkD_backbone(imgs)
        score_map=self.build_score_map(backbone)        
        logits=self.networkD_classifier(backbone)                
        return score_map,logits
        
