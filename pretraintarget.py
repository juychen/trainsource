#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import copy
import os
import sys
import time
from anndata._core.aligned_mapping import V

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pandas.core.arrays import boolean
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import models
import scanpypip.preprocessing as pp
import scanpypip.utils as scut
import utils as ut
from models import AEBase, Predictor, PretrainedPredictor,VAEBase,vae_loss

# class Arguments:
#     def __init__(self):   
#         self.epochs = 500
#         self.bottleneck = 512
#         self.missing_value = np.nan
#         self.target_data = "data/GSE108394/GSM2897334/"
#         self.source_data = "data/GDSC2_expression.csv"
#         self.test_size = 0.2
#         self.valid_size = 0.2
#         self.source_model_path = "saved/models/pretrained_novar.pkl"
#         self.target_model_path = "saved/models/"
#         self.logging_file = "saved/logs/"
#         self.batch_size = 200
#         self.source_h_dims = "2048,1024"
#         self.target_h_dims = "512,256"

#         self.var_genes_disp = 0
#         self.pretrain_path = "saved/models/pretrained_novar.pkl"
#         self.min_n_genes = 0
#         self.max_n_genes = 20000
#         self.min_g = 200
#         self.min_c = 3

        
# args = Arguments()

def run_main(args):

    epochs = args.epochs
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    na = args.missing_value
    data_path = args.target_data
    test_size = args.test_size
    valid_size = args.valid_size
    g_disperson = args.var_genes_disp
    min_n_genes = args.min_n_genes
    max_n_genes = args.max_n_genes
    source_model_path = args.source_model_path
    target_model_path = args.target_model_path 
    log_path = args.logging_file
    batch_size = args.batch_size
    encoder_hdims = args.source_h_dims.split(",")
    encoder_hdims = list(map(int, encoder_hdims))
    source_data_path = args.source_data 
    pretrain = args.pretrain
    model = args.model


    # Misc
    now=time.strftime("%Y-%m-%d-%H-%M-%S")
    log_path = log_path+now+".txt"
    export_name = data_path.replace("/","")

    # If target file not exist, 
    if (os.path.exists(target_model_path)==False):
        target_model_path = target_model_path+"/pretrain_"+export_name+"_"+now+".pkl"


    log=open(log_path,"w")
    sys.stdout=log

    # Load data and preprocessing
    adata = pp.read_sc_file(data_path)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    adata = pp.cal_ncount_ngenes(adata)


    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                jitter=0.4, multi_panel=True,save=export_name)
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

    #Preprocess data by filtering
    adata = pp.receipe_my(adata,l_n_genes=min_n_genes,r_n_genes=max_n_genes,filter_mincells=args.min_c,
                        filter_mingenes=args.min_g,normalize=True,log=True)


    # Select highly variable genes
    sc.pp.highly_variable_genes(adata,min_disp=g_disperson,max_disp=np.inf,max_mean=6)
    sc.pl.highly_variable_genes(adata,save=export_name)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]

    #Prepare to normailize and split target data
    data=adata.X
    mmscaler = preprocessing.MinMaxScaler()
    data = mmscaler.fit_transform(data)

    # Split data to train and valid set
    Xtarget_train, Xtarget_valid = train_test_split(data, test_size=valid_size, random_state=42)
    print(Xtarget_train.shape, Xtarget_valid.shape)


    # Select the device of gpu
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


    # Construct target encoder
    if model == "AE":
        encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
        
        loss_function_e = nn.MSELoss()
    elif model == "VAE":
        encoder = VAEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
        #loss_function_e = encoder.loss_function()

    if torch.cuda.is_available():
        encoder.cuda()

    encoder.to(device)
    optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)


    # Pretrain target encoder
    if(model=="AE"):
        pretrain = str(pretrain)
        encoder,loss_report_en = ut.train_extractor_model(net=encoder,data_loaders=dataloaders_pretrain,
                                    optimizer=optimizer_e,loss_function=loss_function_e,
                                    n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain)
        print("Pretrained finished")
            # Extract feature
        embeddings = encoder.encode(X_allTensor).detach().cpu().numpy()

    elif(model=="VAE"):
        pretrain = str(pretrain)
        encoder,loss_report_en = ut.train_VAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                    optimizer=optimizer_e,
                                    n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain)
        print("Pretrained finished")
        # Extract feature
        results = encoder.encode(X_allTensor)
        embeddings = results[0].detach().cpu().numpy()



    # PCA
    sc.tl.pca(adata, svd_solver='arpack')

    # Add embeddings to the adata package
    adata.obsm["X_AE"] = embeddings

    # Generate neighbor graph
    sc.pp.neighbors(adata, n_neighbors=10,use_rep="X_AE")
    #sc.tl.umap(adata)

    # Use t-sne 
    sc.tl.tsne(adata,use_rep="X_AE")

    # Leiden on the data
    sc.tl.leiden(adata)

    # Plot tsne
    sc.pl.tsne(adata,save=export_name,color=["leiden"])

    # Differenrial expression genes
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False,save=export_name)

    # Save adata
    adata.write("saved/results"+export_name+now+".h5ad")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--source_data', type=str, default='data/GDSC2_expression.csv')
    parser.add_argument('--target_data', type=str, default="data/GSE117872/GSE117872_good_Data_TPM.txt")
    parser.add_argument('--missing_value', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--var_genes_disp', type=float, default=0)
    parser.add_argument('--min_n_genes', type=int, default=0)
    parser.add_argument('--max_n_genes', type=int, default=20000)
    parser.add_argument('--min_g', type=int, default=200)
    parser.add_argument('--min_c', type=int, default=3)

    # train
    parser.add_argument('--source_model_path', type=str, default='saved/models/pretrained_novar.pkl')
    parser.add_argument('--target_model_path', '-p',  type=str, default='saved/models/')
    parser.add_argument('--pretrain', type=str, default='saved/models/pretrain_encoders.pkl')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--model', type=str, default="VAE")

    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--dimreduce', type=str, default="AE")
    parser.add_argument('--predictor', type=str, default="DNN")
    parser.add_argument('--freeze_pretrain', type=int, default=1)
    parser.add_argument('--source_h_dims', type=str, default="2048,1024")
    parser.add_argument('--target_h_dims', type=str, default="512,256")

    # misc
    parser.add_argument('--message', '-m',  type=str, default='')
    parser.add_argument('--output_name', '-n',  type=str, default='')
    parser.add_argument('--logging_file', '-l',  type=str, default='saved/logs/log')

    #
    args, unknown = parser.parse_known_args()
    run_main(args)
