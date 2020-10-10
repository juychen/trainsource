#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
from pandas.core.arrays import boolean
import torch
from scipy import stats
from sklearn import preprocessing
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import models
import utils as ut
from models import AEBase, Predictor, PretrainedPredictor

import scanpypip.preprocessing as pp
import scanpypip.utils as scut 

import scanpy as sc


# In[2]:


class Arguments:
    def __init__(self):   
        self.epochs = 500
        self.bottleneck = 512
        self.missing_value = np.nan
        self.target_data = "data/GSE108394/GSM2897334/"
        self.source_data = "data/GDSC2_expression.csv"
        self.test_size = 0.2
        self.valid_size = 0.2
        self.source_model_path = "saved/models/pretrained_novar.pkl"
        self.target_model_path = "saved/models/"
        self.logging_file = "saved/logs/"
        self.batch_size = 200
        self.source_h_dims = "2048,1024"
        self.target_h_dims = "512,256"

        self.var_genes_disp = 0
        self.pretrain_path = "saved/models/pretrained_novar.pkl"
        self.min_n_genes = 0
        self.max_n_genes = 20000
        self.min_g = 200
        self.min_c = 3

        
args = Arguments()


# In[3]:


epochs = args.epochs
dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
dim_dnn_in = dim_au_out
dim_dnn_out=1
na = args.missing_value
data_path = args.target_data
test_size = args.test_size
valid_size = args.valid_size
g_disperson = args.var_genes_disp
min_n_genes = args.min_n_genes
max_n_genes = args.max_n_genes
source_model_path = args.source_model_path
target_model_path = args.target_model_path 
pretrain_path = args.pretrain_path
log_path = args.logging_file
batch_size = args.batch_size
encoder_hdims = args.source_h_dims.split(",")
encoder_hdims = list(map(int, encoder_hdims))
source_data_path = args.source_data 


# In[4]:


now=time.strftime("%Y-%m-%d-%H-%M-%S")
log_path = log_path+now+".txt"
export_name = data_path.replace("/","")
pretrain_path = "saved/models/ae_"+export_name+now+".pkl"


# In[5]:


#scv = pd.read_csv('data/GSE117872/GSE117872_good_Data_TPM.txt',sep="\t")


# In[6]:


#scv.shape


# In[7]:


adata = pp.read_sc_file('data/GSE117872/GSE117872_good_Data_TPM.txt')


# In[8]:


adata


# In[9]:


# adata = sc.read_10x_mtx(
#  'data/GSE108394/GSM2897334/',  # the directory with the `.mtx` file 
#  var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
#  cache=True)                              # write a cache file for faster subsequent reading


# In[10]:


sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata = pp.cal_ncount_ngenes(adata)


# In[11]:


adata


# In[12]:


sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True,save=export_name)


# In[13]:


sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')


# In[14]:


adata = pp.receipe_my(adata,l_n_genes=min_n_genes,r_n_genes=max_n_genes,filter_mincells=args.min_c,
                      filter_mingenes=args.min_g,normalize=True,log=True)


# In[15]:


sc.pp.highly_variable_genes(adata,min_disp=g_disperson,max_disp=np.inf,max_mean=6)

sc.pl.highly_variable_genes(adata,save=export_name)


# In[16]:


adata.raw = adata

adata = adata[:, adata.var.highly_variable]

data=adata.X


# In[17]:


mmscaler = preprocessing.MinMaxScaler()


# In[18]:


data = mmscaler.fit_transform(data)


# In[19]:


Xtarget_train, Xtarget_valid = train_test_split(data, test_size=valid_size, random_state=42)


# In[20]:


print(Xtarget_train.shape, Xtarget_valid.shape)


# In[21]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
torch.cuda.set_device(device)

# Construct datasets and data loaders
Xtarget_trainTensor = torch.FloatTensor(Xtarget_train).to(device)
Xtarget_validTensor = torch.FloatTensor(Xtarget_valid).to(device)
X_allTensor = torch.FloatTensor(data).to(device)

train_dataset = TensorDataset(Xtarget_trainTensor, Xtarget_trainTensor)
valid_dataset = TensorDataset(Xtarget_validTensor, Xtarget_validTensor)
all_dataset = TensorDataset(X_allTensor, X_allTensor)


Xtarget_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
Xtarget_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

dataloaders_pretrain = {'train':Xtarget_trainDataLoader,'val':Xtarget_validDataLoader}


# In[22]:


len(Xtarget_trainDataLoader)


# In[23]:


encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
#model = VAE(dim_au_in=data_r.shape[1],dim_au_out=128)
if torch.cuda.is_available():
    encoder.cuda()

print(encoder)
encoder.to(device)
optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
loss_function_e = nn.MSELoss()
exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)


# In[24]:


# Read source data
data_r=pd.read_csv(source_data_path,index_col=0)


# In[25]:


source_scaler = preprocessing.MinMaxScaler()
source_data = mmscaler.fit_transform(data_r)


# In[26]:


Xsource_train_all, Xsource_test = train_test_split(source_data, test_size=test_size, random_state=42)
Xsource_train, Xsource_valid = train_test_split(Xsource_train_all, test_size=valid_size, random_state=42)


# In[27]:


# Transform source data
# Construct datasets and data loaders
Xsource_trainTensor = torch.FloatTensor(Xsource_train).to(device)
Xsource_validTensor = torch.FloatTensor(Xsource_valid).to(device)
Xsource_testTensor = torch.FloatTensor(Xsource_test).to(device)
Xsource_allTensor = torch.FloatTensor(source_data).to(device)


sourcetrain_dataset = TensorDataset(Xsource_trainTensor, Xsource_trainTensor)
sourcevalid_dataset = TensorDataset(Xsource_validTensor, Xsource_validTensor)
sourcetest_dataset = TensorDataset(Xsource_testTensor, Xsource_testTensor)
sourceall_dataset = TensorDataset(Xsource_allTensor, Xsource_allTensor)

Xsource_trainDataLoader = DataLoader(dataset=sourcetrain_dataset, batch_size=batch_size, shuffle=True)
Xsource_validDataLoader = DataLoader(dataset=sourcevalid_dataset, batch_size=batch_size, shuffle=True)
X_allDataLoader = DataLoader(dataset=sourceall_dataset, batch_size=batch_size, shuffle=True)

# construct TensorDataset
# trainreducedDataset = TensorDataset(Xsource_trainTensor, Y_trainTensor)
# validreducedDataset = TensorDataset(Xsource_validTensor, Y_validTensor)

# trainDataLoader_p = DataLoader(dataset=trainreducedDataset, batch_size=batch_size, shuffle=True)
# validDataLoader_p = DataLoader(dataset=validreducedDataset, batch_size=batch_size, shuffle=True)

dataloaders_source = {'train':Xsource_trainDataLoader,'val':Xsource_validDataLoader}


# In[28]:


data_r


# In[29]:


# Load source model
source_encoder = AEBase(input_dim=data_r.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
source_encoder.load_state_dict(torch.load(source_model_path))          
source_encoder.to(device)


# In[30]:


# Set discriminator model
discriminator = Predictor(input_dim=dim_au_out,output_dim=2)
discriminator.to(device)
loss_d = nn.CrossEntropyLoss()
optimizer_d = optim.Adam(encoder.parameters(), lr=1e-2)
exp_lr_scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d)


# In[31]:


device


# In[32]:


result = ut.train_transfer_model(source_encoder,encoder,discriminator,
                    dataloaders_source,dataloaders_pretrain,
                    loss_d,loss_function_e,
                    optimizer_e,optimizer_d,
                    exp_lr_scheduler_e,exp_lr_scheduler_d,
                    500,device)


# In[ ]:


# Train target encoder
encoder,loss_report_en = ut.train_extractor_model(net=encoder,data_loaders=dataloaders_pretrain,
                            optimizer=optimizer_e,loss_function=loss_function_e,
                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain_path)

print("Pretrained finished")


# In[ ]:


embeddings = encoder.encode(X_allTensor).detach().cpu().numpy()


# In[ ]:


sc.tl.pca(adata, svd_solver='arpack')


# In[ ]:


adata.obsm["X_AE"] = embeddings


# In[ ]:


sc.pp.neighbors(adata, n_neighbors=10,use_rep="X_AE")
#sc.tl.umap(adata)


# In[ ]:


sc.tl.tsne(adata,use_rep="X_AE")


# In[ ]:


sc.tl.leiden(adata)


# In[ ]:


sc.pl.tsne(adata,save=export_name,color=["leiden"])


# In[ ]:


sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False,save=export_name)


# In[ ]:


adata.write("saved/results"+export_name+".h5ad")

