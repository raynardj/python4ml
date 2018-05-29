import torch
from torch import nn
import math
from torch.nn import functional as F



def conv_rfpad(ni,no,ks,stride=1):
    pad = ks//2
    return nn.Sequential(*[
        nn.ReflectionPad2d([pad]*4),
        nn.Conv2d(ni,no,kernel_size = ks, stride = stride, padding=0, bias = False),
        nn.InstanceNorm2d(no),
        nn.LeakyReLU(inplace = True)
    ])

def conv_layer_ins(ni,no,ks,stride=1):
    return nn.Sequential(*[
        nn.Conv2d(ni,no,kernel_size = ks, stride = stride, padding = (1,1),bias = False),
        nn.InstanceNorm2d(no),
        nn.LeakyReLU(inplace = True)
    ])

def deconv(ni,no,ks,stride=(2,2)):
    return nn.Sequential(*[
        nn.ConvTranspose2d(ni,no,kernel_size = ks, stride = stride, padding = (1,1),bias = False),
        nn.InstanceNorm2d(no),
        nn.LeakyReLU(inplace = True)
    ])
class resblock(nn.Module):
    def __init__(self,fn,ks,shrink=1):
        super(resblock,self).__init__()
        self.conv1 = conv_layer_ins(fn,fn//shrink,ks)
        self.conv2 = conv_layer_ins(fn//shrink,fn,ks)
        
    def forward(self, x):
        x = x+self.conv1(self.conv2(x))
        return x

class generative(nn.Module):
    def __init__(self,bn,fn=256,k_list = None):
        super(generative,self).__init__()
        self.bn = bn # block number list
        self.fn = fn
        self.conv_in = conv_rfpad(3,self.fn//4,7)
        
        # Down sampling
        self.conv_down1 = conv_layer_ins(self.fn//4,self.fn//2,ks=3,stride=2)
        self.conv_down2 = conv_layer_ins(self.fn//2,self.fn,ks=3,stride=2)
        
        for i in range(self.bn):
            setattr(self,"resblock_b%s"%(i),resblock(self.fn,ks=3))
        
        # Upsampling using deconv
        self.deconv_1 = deconv(self.fn,self.fn//2,ks=3)
        self.deconv_2 = deconv(self.fn//2,self.fn//4,ks=3)
        self.conv_out = conv_rfpad(self.fn//4,3,ks=7)
    
    def forward(self,x):
        x = self.conv_in(x)
        x = self.conv_down2(self.conv_down1(x))
        for i in range(self.bn):
            x = getattr(self,"resblock_b%s"%(i))(x)
        x = self.deconv_2(self.deconv_1(x))
        return F.tanh(self.conv_out(x))

class discriminative(nn.Module):
    def __init__(self):
        super(discriminative,self).__init__()
        self.conv_in = conv_layer_ins(3,64,3,stride=2)
        layers = []
        self.convs = nn.Sequential(*[
            conv_layer_ins(64,128,3,stride=2),
            conv_layer_ins(128,256,3,stride=2),
            conv_layer_ins(256,512,3,stride=2),
            nn.Conv2d(512,1,3,stride=1,padding=1,bias=False),
        ])
        
    def forward(self,x):
        bs = x.size()[0]
        x = self.convs(self.conv_in(x))
        return x.view(bs,-1).mean(1)
    
class generative_chimney(nn.Module):
    def __init__(self,fn_list, dsamp = 3,k_list = None, diameter=128):
        super(generative_chimney,self).__init__()
        self.fn_list = fn_list
        self.diameter = diameter
        self.dsamp = dsamp
        if k_list == None:
            self.k_list = [3]*(len(self.fn_list)-1)
        else:
            self.k_list = k_list
        self.leaky = nn.LeakyReLU()
        self.conv_in = nn.Conv2d(3, self.diameter,kernel_size = 3, padding = 1,stride=1,bias = False)
        for i in range(self.dsamp):
            setattr(self,"down_1_%s"%(i),nn.Conv2d(self.diameter, self.diameter,kernel_size = 3, padding = 1,stride=1,bias = False))
            setattr(self,"down_2_%s"%(i),nn.Conv2d(self.diameter, self.diameter,kernel_size = 3, padding = 1,stride=2,bias = False))
            
            setattr(self,"bn_down_1_%s"%(i),nn.BatchNorm2d(self.diameter))
            setattr(self,"bn_down_2_%s"%(i),nn.BatchNorm2d(self.diameter))
        
            setattr(self,"level_1_%s"%(i),nn.Conv2d(self.diameter,self.diameter,kernel_size=3, padding = 1, bias = False))
            setattr(self,"level_2_%s"%(i),nn.Conv2d(self.diameter,self.diameter,kernel_size=3, padding = 1, bias = False))
            setattr(self,"level_3_%s"%(i),nn.Conv2d(self.diameter,self.diameter,kernel_size=3, padding = 1, bias = False))
        
            setattr(self,"bn_level_1_%s"%(i), nn.BatchNorm2d(self.diameter))
            setattr(self,"bn_level_2_%s"%(i), nn.BatchNorm2d(self.diameter))
            setattr(self,"bn_level_3_%s"%(i), nn.BatchNorm2d(self.diameter))
            
        self.upout = nn.Conv2d(self.diameter, self.fn_list[0],kernel_size = 3, padding = 1,stride=1,bias = False)
        
        self.up = nn.Upsample(scale_factor = 2)
        
        for i in range(len(self.fn_list)-1):
            setattr(self,"conv_%s"%(i),nn.Conv2d(self.fn_list[i],
                                                 self.fn_list[i+1],
                                                 kernel_size = self.k_list[i],
                                                 padding = self.k2pad(self.k_list[i]),
                                                 bias = False))
            setattr(self,"conv_bn_%s"%(i),nn.BatchNorm2d(self.fn_list[i+1]))
            
        self.conv_out = nn.Conv2d(self.fn_list[-1],3,1,bias=False)
        self.bn_out = nn.BatchNorm2d(3)
            
    def k2pad(self,k):
        return math.floor(k/2)
    
    def forward(self,x):
        x = self.conv_in(x)
        
        for i in range(self.dsamp):
            x = getattr(self,"down_1_%s"%(i))(x)
            x = getattr(self,"bn_down_1_%s"%(i))(x)
            x = self.leaky(x)
            x = getattr(self,"down_2_%s"%(i))(x)
            x = getattr(self,"bn_down_2_%s"%(i))(x)
            x = self.leaky(x)
            
        for i in range(self.dsamp):
            x = self.up(x)
            
            x = getattr(self,"level_1_%s"%(i))(x)
            x = getattr(self,"bn_level_1_%s"%(i))(x)
            x = self.leaky(x)
            x = getattr(self,"level_2_%s"%(i))(x)
            x = getattr(self,"bn_level_2_%s"%(i))(x)
            x = self.leaky(x)
            x = getattr(self,"level_3_%s"%(i))(x)
            x = getattr(self,"bn_level_3_%s"%(i))(x)
            x = self.leaky(x)
            
        x = self.upout(x)
        x = self.leaky(x)
        
        for i in range(len(self.fn_list)-1):
            x = getattr(self,"conv_%s"%(i))(x)
            x = getattr(self,"conv_bn_%s"%(i))(x)
            x = self.leaky(x)

        x = self.conv_out(x)
        # x = self.bn_out(x)
        x = F.sigmoid(x)
        return x
    
class generative_chimney2(nn.Module):
    def __init__(self,fn_list, dsamp = 1,k_list = None, diameter=128):
        super(generative_chimney2,self).__init__()
        self.fn_list = fn_list
        self.diameter = diameter
        self.dsamp = dsamp
        if k_list == None:
            self.k_list = [5]*(len(self.fn_list)-1)
        else:
            self.k_list = k_list
        self.leaky = nn.LeakyReLU()
        self.conv_in = nn.Conv2d(3, self.diameter,kernel_size = 3, padding = 1,stride=1,bias = False)
        for i in range(self.dsamp):
            setattr(self,"down_1_%s"%(i),nn.Conv2d(self.diameter, self.diameter,kernel_size = 3, padding = 1,stride=1,bias = False))
            setattr(self,"down_2_%s"%(i),nn.Conv2d(self.diameter, self.diameter,kernel_size = 3, padding = 1,stride=2,bias = False))
            
            setattr(self,"bn_down_1_%s"%(i),nn.BatchNorm2d(self.diameter))
            setattr(self,"bn_down_2_%s"%(i),nn.BatchNorm2d(self.diameter))
        
            setattr(self,"level_1_%s"%(i),nn.Conv2d(self.diameter,self.diameter,kernel_size=3, padding = 1, bias = False))
            setattr(self,"level_2_%s"%(i),nn.Conv2d(self.diameter,self.diameter,kernel_size=3, padding = 1, bias = False))
            setattr(self,"level_3_%s"%(i),nn.Conv2d(self.diameter,self.diameter,kernel_size=3, padding = 1, bias = False))
        
            setattr(self,"bn_level_1_%s"%(i), nn.BatchNorm2d(self.diameter))
            setattr(self,"bn_level_2_%s"%(i), nn.BatchNorm2d(self.diameter))
            setattr(self,"bn_level_3_%s"%(i), nn.BatchNorm2d(self.diameter))
            
        self.upout = nn.Conv2d(self.diameter, self.fn_list[0],kernel_size = 3, padding = 1,stride=1,bias = False)
        
        self.up = nn.Upsample(scale_factor = 2)
        
        for i in range(len(self.fn_list)-1):
            setattr(self,"conv_%s"%(i),nn.Conv2d(self.fn_list[i],
                                                 self.fn_list[i+1],
                                                 kernel_size = self.k_list[i],
                                                 padding = self.k2pad(self.k_list[i]),
                                                 bias = False))
            setattr(self,"conv_bn_%s"%(i),nn.BatchNorm2d(self.fn_list[i+1]))
            
        self.conv_out = nn.Conv2d(self.fn_list[-1],3,1,bias=False)
        self.bn_out = nn.BatchNorm2d(3)
            
    def k2pad(self,k):
        return math.floor(k/2)
    
    def forward(self,x):
        x = self.conv_in(x)
        
        for i in range(self.dsamp):
            x = getattr(self,"down_1_%s"%(i))(x)
            x = getattr(self,"bn_down_1_%s"%(i))(x)
            x = self.leaky(x)
            x = getattr(self,"down_2_%s"%(i))(x)
            x = getattr(self,"bn_down_2_%s"%(i))(x)
            x = self.leaky(x)
            
            x = self.up(x)
            
            x = getattr(self,"level_1_%s"%(i))(x)
            x = getattr(self,"bn_level_1_%s"%(i))(x)
            x = self.leaky(x)
            x = getattr(self,"level_2_%s"%(i))(x)
            x = getattr(self,"bn_level_2_%s"%(i))(x)
            x = self.leaky(x)
            x = getattr(self,"level_3_%s"%(i))(x)
            x = getattr(self,"bn_level_3_%s"%(i))(x)
            x = self.leaky(x)
            
        x = self.upout(x)
        x = self.leaky(x)
        
        for i in range(len(self.fn_list)-1):
            x = getattr(self,"conv_%s"%(i))(x)
            x = getattr(self,"conv_bn_%s"%(i))(x)
            x = self.leaky(x)

        x = self.conv_out(x)
        # x = self.bn_out(x)
        x = F.sigmoid(x)
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