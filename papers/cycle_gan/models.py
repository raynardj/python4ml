import torch
from torch import nn
import math

class resblock(nn.Module):
    def __init__(self,out_,k_):
        super(resblock,self).__init__()
        
        self.out_ = out_
        self.k_ = k_
        
        self.padding = math.floor(self.k_/2)
        self.conv1 = nn.Conv2d(self.out_,self.out_,self.k_,stride=2,padding=self.padding,bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_)
        self.leaky1 = nn.LeakyReLU()
        self.upconv = nn.Upsample(scale_factor=2)
        self.bn2 = nn.BatchNorm2d(self.out_)
        
    def forward(self, x):
        x1 = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky1(x)
        
        x = self.upconv(x)
        x = self.bn2(x)
        x = x + x1
        return x
        

class generative(nn.Module):
    def __init__(self,fn_list,k_list = None):
        super(generative,self).__init__()
        self.fn_list = fn_list
        if k_list == None:
            self.k_list = [3]*(len(self.fn_list)-1)
        else:
            self.k_list = k_list
        self.leaky = nn.LeakyReLU()
        for i in range(len(self.fn_list)-1):
            setattr(self,"tran_%s"%(i),nn.Conv2d(self.fn_list[i],
                                                 self.fn_list[i+1],
                                                 kernel_size = self.k_list[i],
                                                 padding = self.k2pad(self.k_list[i]),
                                                 bias = False))
            setattr(self,"bn_trans_%s"%(i),nn.BatchNorm2d(self.fn_list[i+1]))
            
            setattr(self,"resblock_%s"%(i),resblock(self.fn_list[i+1],self.k_list[i]))
            setattr(self,"bn_res_%s"%(i),nn.BatchNorm2d(self.fn_list[i+1]))
            
        self.conv_out = nn.Conv2d(self.fn_list[-1],3,1,bias=False)
            
    def k2pad(self,k):
        return math.floor(k/2)
    
    def forward(self,x):
        for i in range(len(self.fn_list)-1):
            x = getattr(self,"tran_%s"%(i))(x)
            x = getattr(self,"bn_trans_%s"%(i))(x)
            x = self.leaky(x)
            x = getattr(self,"resblock_%s"%(i))(x)
            x = getattr(self,"bn_res_%s"%(i))(x)
            x = self.leaky(x)
        x = self.conv_out(x)
        return x

class resblock_d(nn.Module):
    def __init__(self,in_,out_,k_,downsample=True):
        super(resblock_d,self).__init__()
        
        self.in_ = in_
        self.out_ = out_
        self.k_ = k_
        self.downsample = downsample
        
        self.leaky = nn.LeakyReLU()
        
        self.conv1 = nn.Conv2d(self.in_,
                               self.in_,
                               kernel_size=self.k_,
                               stride=1,
                               padding=self.k2pad(self.k_),
                               bias=False)
        self.bn_1 = nn.BatchNorm2d(self.in_)
        
        self.conv2 = nn.Conv2d(self.in_,
                               self.in_,
                               kernel_size=self.k_,
                               stride=1,
                               padding=self.k2pad(self.k_),
                               bias=False)
        self.bn_2 = nn.BatchNorm2d(self.in_)
        
        if self.downsample:
            self.out = nn.Conv2d(self.in_,self.out_,self.k_,
                                 padding=self.k2pad(self.k_),stride=2,bias=False)
            self.bn_out = nn.BatchNorm2d(self.out_)
        
    def k2pad(self,k):
        return math.floor(k/2)
    
    def forward(self,x):
        x1 = x.clone()
        
        x = self.conv1(x)
        x = self.bn_1(x)
        x = self.leaky(x)
        
        x = self.conv2(x)
        x = self.bn_2(x)
        x = self.leaky(x)
        
        x = x + x1
        
        if self.downsample:
            x = self.out(x)
            x = self.bn_out(x)
            x = self.leaky(x)
            
        return x

class discriminative(nn.Module):
    def __init__(self,fn_list,k_list = None):
        super(discriminative,self).__init__()
        self.fn_list = fn_list
        
        if k_list == None:
            self.k_list = [3]*(len(self.fn_list)-1)
        else:
            self.k_list = k_list
            
        self.conv_in = nn.Conv2d(3,self.fn_list[0],3,padding=1,bias=False)
        self.bn_in = nn.BatchNorm2d(self.fn_list[0])
        
        for i in range(len(self.fn_list)-1):
            setattr(self,"res_%s"%(i),resblock_d(self.fn_list[i],
                                                 self.fn_list[i+1],
                                                 k_ = self.k_list[i]))
        
    def forward(self,x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        
        for i in range(len(self.fn_list)-1):
            x = getattr(self,"res_%s"%(i))(x)
        return x