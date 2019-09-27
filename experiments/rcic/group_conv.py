
# coding: utf-8

# # By Plate Group Convolution Top Layers

# Well, I'm kicking myself when I found out the plate leak 2 days before closing, the leak [an official post](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/102905#latest-624588) explained 2 month ago.
# 
# Hence, let's exploit this in a more machine learning way

# In[1]:


import os
import numpy as np
import pandas as pd
import torch
from fastai.vision import *


# First, no time for starting things from ground zero, 
# we use our best version of EfficientNet b5

# In[2]:


SIZE = 456
SITE = 3 # Site: 1:site1, 2:site2, 3:site1 and 2

LR = 1e-5
BS = 64
EPOCHS = 1
MODEL_TYPE = "b5"
LOAD = True # do we load the trained weights
LOAD_NAME = "/home/hadoop/rcic/ensemble/models/rcic-b5-sz456-bs26-s1-s2-r2"
SAVE_NAME = "rcic-%s-sz%s-bs%s-s%s"%(MODEL_TYPE,SIZE,BS,SITE)+""


# This B5 is from other well established notebook of course

# In[3]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

SEED = 0
seed_everything(SEED)


# In[4]:


from pathlib import Path
print("Loading training dataframe")
DATA = Path("/mnt/disk4/cell/")


# The training data

# In[5]:


train_df = pd.read_csv(DATA/'train.csv')
train_df.head(10)


# Now the siRNA => group mapping

# In[6]:


groups = np.load("groups.npy")


# In[7]:


def generate_df(train_df,sample_num=1):
    train_df['path'] = train_df['experiment'].str.cat(train_df['plate'].astype(str).str.cat(train_df['well'],sep='/'),sep='/Plate') + '_s'+str(sample_num) + '_w'
    train_df["pname"] = train_df.apply(lambda x:x["experiment"]+"-"+x["plate"].__str__(), axis=1)
#     train_df["grp"] = train_df.sirna.apply(lambda x:groups[x])
    train_df = train_df.drop(columns=['id_code','experiment','plate','well']).reindex(columns=['path','sirna',"pname"])
    
    return train_df


# In[8]:


site1_train_df = generate_df(train_df)  
site2_train_df = generate_df(train_df, sample_num=2)

if SITE==1: # only site1
    proc_train_df = site1_train_df 
elif SITE==2 : # only site2
    proc_train_df = site2_train_df
elif SITE==3 :
    proc_train_df = pd.concat([site1_train_df,site2_train_df],axis=0 ).reset_index().drop("index",axis=1)
    proc_train_df.to_csv("train_with_bc.csv")


# ### Conv Model

# In[9]:


from efficientnet_pytorch import *

def efficientnet_multichannel(pretrained=True,name='b5',num_classes=1108,num_channels=6,image_size=SIZE):
    model = EfficientNet.from_pretrained('efficientnet-'+name,num_classes=num_classes)
    #model.load_state_dict(torch.load(EFFICIENTNET_MODELS[name]))
    w = model._conv_stem.weight
    #s = model._conv_stem.static_padding
    model._conv_stem = utils.Conv2dStaticSamePadding(num_channels,32,kernel_size=(3, 3), stride=(2, 2), bias=False, image_size = image_size)
    model._conv_stem.weight = nn.Parameter(torch.stack([torch.mean(w, 1)]*num_channels, dim=1))
    return model


# In[10]:


model = efficientnet_multichannel(pretrained=False if LOAD else True,name=MODEL_TYPE)


# In[11]:


loaded_dict = torch.load(LOAD_NAME+".pth", map_location={'cuda:0': 'cpu'})


# In[12]:


if LOAD:
    model.eval()
    model.load_state_dict(loaded_dict["model"])
    model.eval()


# ### Dataset

# In[13]:


from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2


# In[14]:


def open_rcic_image(fn):
    images = []
    for i in range(6):
        file_name = fn+str(i+1)+'.png'
        im = cv2.imread(file_name)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)
    image = np.dstack(images)
    #print(pil2tensor(image, np.float32).shape)#.div_(255).shape)
    return Image(pil2tensor(image, np.float32).div_(255))

class MultiChannelImageList(ImageList):
    def open(self, fn):
        return open_rcic_image(fn)


# In[15]:


data = (MultiChannelImageList.from_df(df=proc_train_df,path=DATA/'train/')
        .split_none()
        .label_from_df()
        .transform(get_transforms(),size=SIZE)
        .databunch(bs=BS,num_workers=4)
        .normalize()
       )


# In[16]:


data.train_dl = data.train_dl.new(shuffle=False)


# In[17]:


delattr(model,"_fc")


# In[18]:


class modelConv(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.md = model
        
    def forward(self,x):
        x = self.md.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        return x


# In[19]:


mc = modelConv(model)

learn = Learner(data, mc, metrics=[accuracy]).to_fp16()

learn.path = Path('/data/rcic')


# In[ ]:


import bcolz as bz


# In[45]:


# !rm -rf /data/rcic/actv_train


# In[21]:


barr= bz.carray(np.zeros((0,2048)),rootdir="/data/rcic/actv_train")


# In[ ]:


print("Total batch len:",len(learn.data.train_dl))
gen = iter(learn.data.train_dl)
for i in range(len(learn.data.train_dl)):
# for i in range(2):
    x,y = next(gen)
    p = learn.pred_batch(batch=(x.cuda().half(),y))
    barr.append(p.numpy())
    barr.flush()
    sys.stdout.write("\r batch [%s] saved"%(i))


# In[31]:


test_df = pd.read_csv(DATA/"test.csv")


# In[38]:


proc_test_df1 = generate_df(test_df.copy(),sample_num= 1)
data_test1 = MultiChannelImageList.from_df(df=proc_test_df1,path=DATA/'test/')
proc_test_df2 = generate_df(test_df.copy(),sample_num= 2)
data_test2 = MultiChannelImageList.from_df(df=proc_test_df2,path=DATA/'test/')


# In[43]:


learn.data.add_test(data_test1)


# In[ ]:


barrt1= bz.carray(np.zeros((0,2048)),rootdir="/data/rcic/actv_test1")


# In[44]:


print("Total batch len:",len(learn.data.test_dl))

gen = iter(learn.data.test_dl)
for i in range(len(learn.data.test_dl)):
# for i in range(2):
    x,y = next(gen)
    p = learn.pred_batch(batch=(x.cuda().half(),y))
    barrt1.append(p.numpy())
    barrt1.flush()
    sys.stdout.write("\r batch [%s] saved"%(i))
    
print("test 1 finished")


# In[41]:


learn.data.add_test(data_test2)


# In[42]:


barrt2= bz.carray(np.zeros((0,2048)),rootdir="/data/rcic/actv_test2")


# In[ ]:


print("Total batch len:",len(learn.data.test_dl))

gen = iter(learn.data.test_dl)
for i in range(len(learn.data.test_dl)):
# for i in range(2):
    x,y = next(gen)
    p = learn.pred_batch(batch=(x.cuda().half(),y))
    barrt2.append(p.numpy())
    barrt2.flush()
    sys.stdout.write("\r batch [%s] saved"%(i))


# In[ ]:


print("test 2 finished")

