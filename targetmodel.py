
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
from sklearn.metrics import r2_score
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import models
import utils as ut
from models import AEBase, Predictor, PretrainedPredictor
import scanpy as sc
import scanpypip.preprocessing as pp

#import scipy.io as sio



# Define parameters
epochs = 500 #200,500,1000
dim_au_in = 11833
dim_au_out = 512 #8, 16, 32, 64, 128, 256,512
dim_dnn_in = dim_au_out
dim_dnn_out=1

# Edit in 2020 09 21 main function
def run_main(args):

    # Define parameters
    epochs = args.epochs
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    dim_dnn_in = dim_au_out
    dim_dnn_out=1
    na = args.missing_value
    data_path = args.data_path
    test_size = args.test_size
    valid_size = args.valid_size
    g_disperson = args.var_genes_disp
    min_n_genes = args.min_n_genes
    max_n_genes = args.max_n_genes
    model_path = args.model_store_path
    pretrain_path = args.pretrain_path
    log_path = args.logging_file
    batch_size = args.batch_size
    encoder_hdims = args.ft_h_dims.split(",")
    encoder_hdims = list(map(int, encoder_hdims))
    print(args)

    # Misc
    now=time.strftime("%Y-%m-%d-%H-%M-%S")
    log_path = log_path+now+".txt"
    export_name = data_path.replace("/","")
    pretrain_path = "saved/models/ae_"+export_name+now+".pkl"

    log_path = log_path+now+".txt"
    log=open(log_path,"w")
    sys.stdout=log


    # Read data
    adata = sc.read_10x_mtx(
    'data/GSE108394/GSM2897334/',  # the directory with the `.mtx` file 
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)                              # write a cache file for faster subsequent reading



    # data = data_r
    adata = pp.receipe_my(adata,l_n_genes=min_n_genes,r_n_genes=max_n_genes,filter_mincells=args.min_c,
                        filter_mingenes=args.min_g,normalize=True,log=True)

    # Save qc metrics
    sc.pl.violin(adata, ['n_counts',"percent_mito",'percent_rps', 'percent_rpl'],
             jitter=0.4, multi_panel=True,save=export_name)

    # HVG

    sc.pp.highly_variable_genes(adata,min_disp=g_disperson,max_disp=np.inf)

    sc.pl.highly_variable_genes(adata,save=export_name)

    # Extract data
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    data=adata.X
            
    # Scaling and splitting data
    mmscaler = preprocessing.MinMaxScaler()
    data = mmscaler.fit_transform(data.todense())
    X_train, X_valid = train_test_split(data, test_size=valid_size, random_state=42)
    print(X_train.shape, X_valid.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    torch.cuda.set_device(device)

    # Construct datasets and data loaders
    X_trainTensor = torch.FloatTensor(X_train).to(device)
    X_validTensor = torch.FloatTensor(X_valid).to(device)
    X_allTensor = torch.FloatTensor(data).to(device)

    train_dataset = TensorDataset(X_trainTensor, X_trainTensor)
    valid_dataset = TensorDataset(X_validTensor, X_validTensor)
    all_dataset = TensorDataset(X_allTensor, X_allTensor)


    X_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    X_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    dataloaders_pretrain = {'train':X_trainDataLoader,'val':X_validDataLoader}

    # Traing models
    encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
    #model = VAE(dim_au_in=data_r.shape[1],dim_au_out=128)
    if torch.cuda.is_available():
        encoder.cuda()

    print(encoder)
    encoder.to(device)
    optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
    loss_function_e = nn.MSELoss()
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)
    encoder,loss_report_en = ut.train_extractor_model(net=encoder,data_loaders=dataloaders_pretrain,
                                optimizer=optimizer_e,loss_function=loss_function_e,
                                n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain_path)


    print("Pretrained finished")

    # Extract embeddings
    embeddings = encoder.encode(X_allTensor).detach().cpu().numpy()

    # Process embeddings 
    sc.tl.pca(adata, svd_solver='arpack')
    adata.obsm["X_AE"] = embeddings

    # Visualize embeddings
    sc.tl.tsne(adata,use_rep="X_AE")
    
    # Clustering
    sc.pp.neighbors(adata, n_neighbors=10,use_rep="X_AE")
    sc.tl.leiden(adata)

    # Plot tsne 
    sc.pl.tsne(adata,save=export_name,color=["leiden"])

    # Print ``
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False,save=export_name)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--data_path', type=str, default='data/GDSC2_expression.csv')
    parser.add_argument('--missing_value', type=int, default=1)
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--var_genes_disp', type=float, default=None)
    parser.add_argument('--min_n_genes', type=int, default=0)
    parser.add_argument('--max_n_genes', type=int, default=20000)
    parser.add_argument('--min_g', type=int, default=200)
    parser.add_argument('--min_c', type=int, default=3)

    parser.add_argument('--var_genes_disp', type=float, default=None)


    # train
    parser.add_argument('--pretrain_path', type=str, default='saved/models/pretrained.pkl')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--ft_h_dims', type=str, default="2048,1024")


    # misc
    parser.add_argument('--message', '-m',  type=str, default='')
    parser.add_argument('--output_name', '-n',  type=str, default='')
    parser.add_argument('--model_store_path', '-p',  type=str, default='saved/models/model.pkl')
    parser.add_argument('--logging_file', '-l',  type=str, default='saved/logs/log')

    #
    args, unknown = parser.parse_known_args()
    run_main(args)

