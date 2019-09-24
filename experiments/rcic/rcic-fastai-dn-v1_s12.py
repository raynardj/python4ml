
# coding: utf-8

# # Recursion Cellular Image Classification - Densenet 
# 
# Thanks greatly to [this kaggle kernel](https://www.kaggle.com/kernels/scriptcontent/20557703/download)

# ## Load modules

# In[1]:


import os
import numpy as np
import pandas as pd
import torch
from fastai.vision import *


# In[2]:


print("GPU cuda:\t",torch.cuda.is_available())


# In[3]:


SIZE = 448
SITE = 3 # Site: 1:site1, 2:site2, 3:site1 and 2
BS = 26
EPOCHS = 18
MODEL_TYPE = "dn121"
LOAD = False # do we load the trained weights
LOAD_NAME = "rcic____"
SAVE_NAME = "rcic-%s-sz%s-bs%s-s%s"%(MODEL_TYPE,SIZE,BS,SITE)+""
USE_PSEUDO = False # use pseudo labeling
SUB = Path("/home/hadoop/github/python4ml/experiments/rcic/submissions/") # submit directory
PSEUDO = SUB/"submission_ens_3b3_2b4.csv"
print("learn save name",SAVE_NAME)


# In[4]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

SEED = 0
seed_everything(SEED)



# ## Loading and formatting data
# 
# Here I will load the csv into the DataFrame, and create a column in the DataFrame with the path to the corresponding image (`generate_df`)

# In[5]:


from pathlib import Path
print("Loading training dataframe")
DATA = Path("/mnt/disk4/cell/")


# In[6]:


train_df = pd.read_csv(DATA/'train.csv')
train_df.head(10)


# In[7]:


if USE_PSEUDO:
    pseudo = pd.read_csv(PSEUDO)
    test_df = pd.read_csv(DATA/'test.csv')
    pseudo = pseudo.set_index("id_code").join(test_df.set_index("id_code"), how="inner").reset_index()
    pseudo = pseudo[train_df.columns]

    train_df["folder"] = "train"
    pseudo["folder"] = "test"
    
    split__ = np.random.rand(len(pseudo))>0.5

    pseudo1 = pseudo[split__]
    pseudo2 = pseudo[~split__]
                 
    dirty_df_1 = pd.concat([train_df,pseudo1]).sample(frac=1.).reset_index().drop("index",axis=1)
    dirty_df_2 = pd.concat([train_df,pseudo2]).sample(frac=1.).reset_index().drop("index",axis=1)


# In[8]:


def generate_df(train_df,sample_num=1):
    train_df['path'] = train_df['experiment'].str.cat(train_df['plate'].astype(str).str.cat(train_df['well'],sep='/'),sep='/Plate') + '_s'+str(sample_num) + '_w'
    train_df = train_df.drop(columns=['id_code','experiment','plate','well']).reindex(columns=['path','sirna'])
    return train_df

def generate_df_pseudo(train_df,sample_num=1):
    train_df['path'] = train_df["folder"].str.cat(train_df['experiment'].str.cat(train_df['plate'].astype(str).str.cat(train_df['well'],sep='/'),sep='/Plate'),sep="/") + '_s'+str(sample_num) + '_w'
    train_df = train_df.drop(columns=['id_code','experiment','plate','well']).reindex(columns=['path','sirna'])
    return train_df
if USE_PSEUDO:
    site1_train_df = generate_df_pseudo(dirty_df_1)  
    site2_train_df = generate_df_pseudo(dirty_df_2)
else:
    site1_train_df = generate_df(train_df)  
    site2_train_df = generate_df(train_df, sample_num=2)


if SITE==1: # only site1
    proc_train_df = site1_train_df 
elif SITE==2 : # only site2
    proc_train_df = site2_train_df
elif SITE==3 :
    proc_train_df = pd.concat([site1_train_df,site2_train_df],axis=0 )    .sample(frac = 1.0)    .reset_index()    .drop("index",axis=1)


# In[9]:


proc_train_df.head(10)


# Let's look at an example image. These images are 6-channel images, but the each of the six channels are saved as separate files. Here, I open just one channel of the image.

# In[10]:


import cv2
img = cv2.imread(str(DATA/"train/HEPG2-01/Plate1/B03_s1_w2.png"))
# plt.imshow(img)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# plt.imshow(gray_img)
gray_img.shape


# In fastai, there is a modular data API that allows you to easily load images, add labels, split into train/valid, and add transforms. The base class for loading the images is an `ItemList`. For image classification tasks, the base class is `ImageList` which in turn subclasses the `ItemList` class. Since `ImageList` can only open 3-channel images, we will define a new `ImageList` class where we redefine the loading function:

# In[11]:


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


# As I subclassed the ImageList function I can load images with the `ImageList` function `.from_df`. 

# In[12]:


il = MultiChannelImageList.from_df(df=proc_train_df,path=DATA/'train/')


# We have to redefine the following function to be able to view the image in the notebook. I view just the first 3 channels.

# In[13]:


def image2np(image:Tensor)->np.ndarray:
    "Convert from torch style `image` to numpy/matplotlib style."
    res = image.cpu().permute(1,2,0).numpy()
    if res.shape[2]==1:
        return res[...,0]  
    elif res.shape[2]>3:
        #print(res.shape)
        #print(res[...,:3].shape)
        return res[...,:3]
    else:
        return res

vision.image.image2np = image2np


# Now let's view an example image:

# In[14]:


# il[0]


# With the multi-channel `ImageList` defined, we can now create a DataBunch of the train images. Let's first create a stratified split of dataset and get the indices. 

# In[15]:


from sklearn.model_selection import StratifiedKFold
#train_idx, val_idx = next(iter(StratifiedKFold(n_splits=int(1/0.035),random_state=42).split(proc_train_df, proc_train_df.sirna)))
from sklearn.model_selection import train_test_split
train_df,val_df = train_test_split(proc_train_df,test_size=0.035, stratify = proc_train_df.sirna, random_state=42)
_proc_train_df = pd.concat([train_df,val_df])


# Now we create the `DataBunch`

# In[16]:


print("Creating data bunch")
data = (MultiChannelImageList.from_df(df=_proc_train_df,path=DATA if USE_PSEUDO else DATA/'train/' )
        .split_by_idx(list(range(len(train_df),len(_proc_train_df))))
        .label_from_df()
        .transform(get_transforms(do_flip=True,flip_vert=True),size=SIZE)
        .databunch(bs=BS,num_workers=4)
        .normalize()
       )


# In[17]:


# data.show_batch()


# ## Creating and Training a Model

# I will use a pretrained EfficientNet. There is code for other models thatt you can try but the EfficientNet seems to do the best. I have to now adjust the CNN arch to take in 6 channels as opposed to the usual 3 channels:

# In[18]:


# !pip install efficientnet_pytorch


# In[19]:


from efficientnet_pytorch import *


# In[20]:


"""Inspired by https://github.com/wdhorton/protein-atlas-fastai/blob/master/resnet.py"""

import torchvision
RESNET_MODELS = {
    18: torchvision.models.resnet18,
    34: torchvision.models.resnet34,
    50: torchvision.models.resnet50,
    101: torchvision.models.resnet101,
    152: torchvision.models.resnet152,
}

def resnet_multichannel(depth=50,pretrained=True,num_classes=1108,num_channels=6):
        model = RESNET_MODELS[depth](pretrained=pretrained)
        w = model.conv1.weight
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        model.conv1.weight = nn.Parameter(torch.stack([torch.mean(w, 1)]*num_channels, dim=1))
        return model

    
DENSENET_MODELS = {
    121: torchvision.models.densenet121,
    161: torchvision.models.densenet161,
    169: torchvision.models.densenet169,
    201: torchvision.models.densenet201,
}



def densenet_multichannel(depth=121,pretrained=True,num_classes=1108,num_channels=6):
    model = DENSENET_MODELS[depth](pretrained=pretrained)
    
    w = model.features.conv0.weight
    model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    model.features.conv0.weight = nn.Parameter(torch.stack([torch.mean(w, 1)]*num_channels, dim=1))
    
    class dnNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = model.features
            self.classifier = nn.Linear(model.classifier.weight.data.size(1), num_classes)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):nn.init.kaiming_normal(m.weight.data)
                elif isinstance(m, nn.BatchNorm2d):m.weight.data.fill_(1);m.bias.data.zero_()
                elif isinstance(m, nn.Linear):m.bias.data.zero_()
            
        def forward(self,x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.avg_pool2d(out, kernel_size=14, stride=1).view(features.size(0), -1)
            out = self.classifier(out)
            return out
    dn_model = dnNet()
    return dn_model
        
        
#EFFICIENTNET_MODELS = {
#    'b0': '../input/efficientnet-pytorch/efficientnet-b0-08094119.pth',
#    'b1': '../input/efficientnet-pytorch/efficientnet-b1-dbc7070a.pth',
#    'b2': '../input/efficientnet-pytorch/efficientnet-b2-27687264.pth',
#    'b3': '../input/efficientnet-pytorch/efficientnet-b3-c8376fa2.pth',
#    'b4': '../input/efficientnet-pytorch/efficientnet-b4-e116e8b3.pth',
#    'b5': '../input/efficientnet-pytorch/efficientnet-b5-586e6cc6.pth'
#}


def efficientnet_multichannel(pretrained=True,name='b5',num_classes=1108,num_channels=6,image_size=SIZE):
    model = EfficientNet.from_pretrained('efficientnet-'+name,num_classes=num_classes)
    #model.load_state_dict(torch.load(EFFICIENTNET_MODELS[name]))
    w = model._conv_stem.weight
    #s = model._conv_stem.static_padding
    model._conv_stem = utils.Conv2dStaticSamePadding(num_channels,32,kernel_size=(3, 3), stride=(2, 2), bias=False, image_size = image_size)
    model._conv_stem.weight = nn.Parameter(torch.stack([torch.mean(w, 1)]*num_channels, dim=1))
    return model


# In[21]:


def resnet18(pretrained,num_channels=6):
    return resnet_multichannel(depth=18,pretrained=pretrained,num_channels=num_channels)

def _resnet_split(m): return (m[0][6],m[1])

def densenet(name,pretrained=True,num_channels=6):
    return densenet_multichannel(depth=int(name.replace("dn","")),pretrained=pretrained,num_channels=num_channels)
  
def _densenet_split(m:nn.Module): return (m[0][0][7],m[1])

def efficientnetbn(name = "b3", pretrained=True,num_channels=6):
    return efficientnet_multichannel(pretrained=pretrained,name=name,num_channels=num_channels)

def getModel(name):
    if name in list("b%s"%(i) for i in range(6)):
        return efficientnetbn(name = name)
    elif name in ["dn121","dn169","dn161","dn201"]:
        return densenet(name)


# Let's create our Learner:

# In[22]:


from fastai.metrics import *
print("Create model")
learn = Learner(data, getModel(MODEL_TYPE),metrics=[accuracy]).to_fp16()
learn.path = Path('/data/rcic')


# We will now unfreeze and train the entire model.

# In[23]:


if LOAD:
    learn.load(LOAD_NAME)


# In[24]:


learn.unfreeze()
#learn.lr_find() #<-- uncomment to determine the learning rate (commented to reduce time)
#learn.recorder.plot(suggestion=True) 


# In[25]:


from fastai.callbacks import SaveModelCallback


# In[ ]:


learn.fit_one_cycle(EPOCHS,1e-3,callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy')])


# In[ ]:


# learn.recorder.plot_losses()
# learn.recorder.plot_metrics()


# In[ ]:


learn.save(SAVE_NAME)
learn.export()


# ## Inference and Submission Generation

# Let's now load our test csv and process the DataFrame like we did for the training data.

# In[ ]:


test_df = pd.read_csv(DATA/'test.csv')
proc_test_df = generate_df(test_df.copy(),sample_num= 2 if SITE==2 else 1)


# We add the data to our DataBunch:

# In[ ]:


data_test = MultiChannelImageList.from_df(df=proc_test_df,path=DATA/'test/')
learn.data.add_test(data_test)


# Now we can get out predictions on the test set.

# In[ ]:


preds, _ = learn.get_preds(DatasetType.Test)


# In[ ]:


preds_ = preds.argmax(dim=-1)


# In[ ]:


test_df.head(10)


# Let's open the sample submission file and load it with our predictions to create a submission.

# In[ ]:


submission_df = pd.read_csv(DATA/'sample_submission.csv')


# In[ ]:


submission_df.sirna = preds_.numpy().astype(int)
print(submission_df.head(5))


# In[ ]:


submission_df.to_csv('submission.csv',index=False)

