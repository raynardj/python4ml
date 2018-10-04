# coding: utf-8

# ## 20Mn data MovieLens Experiment

# In[1]:


import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam


# In[2]:


from ray.lprint import lprint
l = lprint("experiment with RNN+CF on movielens 20m data")


# In[3]:


CUDA = torch.cuda.is_available()
SEQ_LEN = 19
DIM = 100
l.p("has GPU cuda",CUDA)


# In[4]:


# %ls /data/ml-20m


# In[5]:


DATA = "/data/ml-20m/ratings.csv"


# In[6]:


l.p("loading csv file", DATA)
rate_df = pd.read_csv(DATA)
l.p("csv file loaded")


# In[7]:


len(rate_df)


# In[8]:


rate_df.groupby("userId").count()["movieId"].min()
# The minimum number of movies a user have rated


# In[9]:


userId = list(set(rate_df["userId"]))
movieId = list(set(rate_df["movieId"]))
print("total number of users and movies:\t",len(userId),"\t",len(movieId))


# In[10]:


l.p("making dictionary")
u2i = dict((v,k) for k,v in enumerate(userId))
m2i = dict((v,k) for k,v in enumerate(movieId))
i2u = dict((k,v) for k,v in enumerate(userId))
i2m = dict((k,v) for k,v in enumerate(movieId))


# In[11]:


# Translating original index to the new index
rate_df["movieIdx"] = rate_df.movieId.apply(lambda x:m2i[x]).astype(int)
rate_df["userIdx"] = rate_df.userId.apply(lambda x:u2i[x]).astype(int)
rate_df["rating"] = rate_df["rating"]/5


# ### Train /Valid Split: K fold Validation 

# In[12]:


l.p("generating groubby slice")
def get_user_trail(rate_df):
    return rate_df.sort_values(by=["userId","timestamp"]).groupby("userId")
    #gb.apply(lambda x:x.sample(n = 20, replace = False))
gb = get_user_trail(rate_df)


# In[13]:


from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import math


# In[14]:


KEEP_CONSEQ = True

if KEEP_CONSEQ:
    # keep the consequtivity among the items the user has rated
    def sample_split(x):
        sample_idx = math.floor(np.random.rand()*(len(x) - SEQ_LEN - 1))
        seq_and_y = x[sample_idx:sample_idx + SEQ_LEN+1]
        return seq_and_y
else:
    # randomly pick the right amount of sample from user's record
    pick_k = np.array([0]*SEQ_LEN +[1])==1

    def sample_split(x):
        sampled = x.sample(n = 20, replace = False)
        seq = sampled.head(19).sort_values(by="timestamp")
        y = sampled[pick_k]
        return pd.concat([seq,y])

class rnn_record(Dataset):
    def __init__(self, gb):
        self.gb = gb
        self.make_seq()
    
    def make_seq(self):
        """
        Resample the data
        """
        self.all_seq = self.gb.apply(sample_split)
        
    def __len__(self):
        return len(self.gb)
        
    def __getitem__(self,idx):
        df = self.all_seq.loc[idx]
        seq = df.head(SEQ_LEN)[["movieIdx","rating"]].values
        targ = df.head(SEQ_LEN+1).tail(1)[["movieIdx","rating"]].values
        targ_v, targ_y =targ[:,0], targ[:,1]
        return idx,seq,targ_v,targ_y


# In[15]:


# Testing data generator

# data_gb = get_user_trail(rate_df)
# rr = rnn_record(data_gb)
# rr.all_seq

# dl = DataLoader(rr,shuffle=True,batch_size=1)
# gen = iter(dl)
# next(gen)


# In[16]:


### Model

class mLinkNet(nn.Module):
    def __init__(self, hidden_size,v_size):
        """
        mLinkNet, short for missing link net
        """
        super(mLinkNet,self).__init__()
        self.hidden_size = hidden_size
        self.v_size = v_size
        self.emb = nn.Embedding(v_size,hidden_size)
        
        self.rnn = nn.GRU(input_size = self.hidden_size+1,
                          hidden_size= hidden_size+1,
                          num_layers=1,
                          batch_first = True,
                          dropout=0)
        
        self.mlp = nn.Sequential(*[
            nn.Dropout(.3),
            nn.Linear(hidden_size*2 + 1, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Linear(256,1,bias=False),
            nn.Sigmoid(),
        ])
    
    def forward(self,seq,targ_v):
        seq_vec = torch.cat([self.emb(seq[:,0].long()),
                             seq[:,1].unsqueeze(-1).float()], dim=2)
        output, hn = self.rnn(seq_vec)
        x = torch.cat([hn.squeeze(0),self.emb(targ_v.long()).squeeze(1)],dim=1)
        return self.mlp(x)


# In[ ]:



def action(*args,**kwargs):
    # get data from data feeder
    idx,seq,targ_v,y = args[0]
    if CUDA:
        seq,targ_v,y = seq.cuda(),targ_v.cuda(),y.cuda()
    y = y.float()
    
    # Clear the Jacobian Matrix
    opt.zero_grad()
    
    # Predict y hat
    y_ = mln(seq, targ_v)
    # Calculate Loss
    loss = loss_func(y_,y)
    
    # Backward Propagation
    loss.backward()
    opt.step()
    # Mean Absolute Loss as print out metrics
    mae = torch.mean(torch.abs(y_-y))
    if kwargs["ite"] == train_len - 1: # resample the sequence
        trainer.train_data.dataset.make_seq()
    return {"loss":loss.item(),"mae":mae.item()}

def val_action(*args,**kwargs):
    """
    A validation step
    Exactly the same like train step, but no learning, only forward pass
    """
    idx,seq,targ_v,y = args[0]
    if CUDA:
        seq,targ_v,y = seq.cuda(),targ_v.cuda(),y.cuda()
    y = y.float()
    
    y_ = mln(seq, targ_v)
    
    loss = loss_func(y_,y)
    mae = torch.mean(torch.abs(y_-y))
    if kwargs["ite"] == valid_len - 1:
        torch.save(mln.state_dict(),"/data/rnn_cf_0.0.2.npy")
        trainer.val_data.dataset.make_seq()
    return {"loss":loss.item(),"mae":mae.item()}


# In[ ]:


l.p("making train/test split")
user_count = len(userId)
K = 3
valid_split = dict({})
random = np.random.rand(user_count)
from ray.matchbox import Trainer

l.p("start training")
for fold in range(K):
    valid_split = ((fold/K) < random)*(random <= ((fold+1)/K))
    train_idx = np.array(range(user_count))[~valid_split]
    valid_idx = np.array(range(user_count))[valid_split]

    train_df = rate_df[rate_df.userId.isin(train_idx)]
    valid_df = rate_df[rate_df.userId.isin(valid_idx)]
    
    # Since user id mapping doesn't matter any more.
    # It's easier to make a dataset with contineous user_id.
    train_u2i = dict((v,k) for k,v in enumerate(set(train_df.userId)))
    valid_u2i = dict((v,k) for k,v in enumerate(set(valid_df.userId)))
    train_df["userId"] = train_df.userId.apply(lambda x:train_u2i[x])
    valid_df["userId"] = valid_df.userId.apply(lambda x:valid_u2i[x])
    
    train_gb = get_user_trail(train_df)
    valid_gb = get_user_trail(valid_df)
    # ds = rnn_record(gb)
    l.p("generating dataset","train")
    train_ds = rnn_record(train_gb)
    l.p("generating dataset","valid")
    valid_ds = rnn_record(valid_gb)
    l.p("dataset generated")

    l.p("creating model")
    mln = mLinkNet(hidden_size = DIM, 
               v_size = len(movieId))
    if CUDA:
        l.p("loading model to GPU")
        torch.cuda.empty_cache()
        mln.cuda()

    opt = Adam(mln.parameters())
    loss_func = nn.MSELoss()
    trainer = Trainer(train_ds, val_dataset=valid_ds, batch_size=16, print_on=3)
    train_len = len(trainer.train_data)
    valid_len = len(trainer.val_data)
    l.p("train_len",train_len)
    l.p("valid_len",valid_len)
    trainer.action  = action
    trainer.val_action  = val_action
    
    l.p("fold training start", fold)
    trainer.train(12,name="rnn_cf_fold%s"%(fold))
    l.p("fold training finished",fold)
l.p("training finished")



