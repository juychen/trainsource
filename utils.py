import copy
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch import device, nn, optim, t
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, dataset
from tqdm import tqdm

import graph_function as g
from gae.model import GCNModelAE, GCNModelVAE, g_loss_function
from gae.utils import get_roc_score, mask_test_edges, preprocess_graph
from models import vae_loss
import re

def highly_variable_genes(data, 
    layer=None, n_top_genes=None, 
    min_disp=0.5, max_disp=np.inf, min_mean=0.0125, max_mean=3, 
    span=0.3, n_bins=20, flavor='seurat', subset=False, inplace=True, batch_key=None, PCA_graph=False, PCA_dim = 50, k = 10, n_pcs=40):

    adata = sc.AnnData(data)

    adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`
    adata.obs_names_make_unique()


    if n_top_genes!=None:
        sc.pp.highly_variable_genes(adata,layer=layer,n_top_genes=n_top_genes,
        span=span, n_bins=n_bins, flavor='seurat_v3', subset=subset, inplace=inplace, batch_key=batch_key)

    else: 
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata,
        layer=layer,n_top_genes=n_top_genes,
        min_disp=min_disp, max_disp=max_disp, min_mean=min_mean, max_mean=max_mean, 
        span=span, n_bins=n_bins, flavor=flavor, subset=subset, inplace=inplace, batch_key=batch_key)

    if PCA_graph == True:
        sc.tl.pca(adata,n_comps=PCA_dim)
        X_pca = adata.obsm["X_pca"]
        sc.pp.neighbors(adata, n_neighbors=k, n_pcs=n_pcs)

        return adata.var.highly_variable,adata,X_pca


    return adata.var.highly_variable,adata

def save_arguments(args,now):
    args_strings =re.sub("\'|\"|Namespace|\(|\)","",str(args)).split(sep=', ')
    args_dict = dict()
    for item in args_strings:
        items = item.split(sep='=')
        args_dict[items[0]] = items[1]

    args_df = pd.DataFrame(args_dict,index=[now]).T
    args_df.to_csv("saved/logs/arguments_" +now + '.csv')

    return args_df

def plot_label_hist(Y,save=None):

    # the histogram of the data
    n, bins, patches = plt.hist(Y, 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('Y values')
    plt.ylabel('Probability')
    plt.title('Histogram of target')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    # plt.grid(True)
    if save == None:
        plt.show()
    else:
        plt.savefig(save)

# plot no skill and model roc curves
def plot_roc_curve(test_y,naive_probs,model_probs,title="",path="figures/roc_curve.pdf"):

    # plot naive skill roc curve
    fpr, tpr, _ = roc_curve(test_y, naive_probs)
    plt.plot(fpr, tpr, linestyle='--', label='Random')
    # plot model roc curve
    fpr, tpr, _ = roc_curve(test_y, model_probs)
    plt.plot(fpr, tpr, marker='.', label='Predition')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.title(title)

    # show the plot
    if path == None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close() 

# plot no skill and model precision-recall curves
def plot_pr_curve(test_y,model_probs,selected_label = 1,title="",path="figures/prc_curve.pdf"):
    # calculate the no skill line as the proportion of the positive class
    no_skill = len(test_y[test_y==selected_label]) / len(test_y)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Random')
    # plot model precision-recall curve
    precision, recall, _ = precision_recall_curve(test_y, model_probs)
    plt.plot(recall, precision, marker='.', label='Predition')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.title(title)

    # show the plot
    if path == None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close() 


def specific_process(adata,dataname="",**kargs):
    if dataname =="GSE117872":
        select_origin = kargs['select_origin']
        adata = process_117872(adata,select_origin=select_origin)

    return adata

def process_117872(adata,**kargs):

    annotation = pd.read_csv('data/GSE117872/GSE117872_good_Data_cellinfo.txt',sep="\t",index_col="groups")
    for item in annotation.columns:
        #adata.obs[str(item)] = annotation.loc[:,item].convert_dtypes('category').values
        adata.obs[str(item)] = annotation.loc[:,item].astype("category")

    if "select_origin" in kargs:
        origin = kargs['select_origin']
        if origin=="all":
            return adata
            
        selected=adata.obs['origin']==origin
        selected=selected.to_numpy('bool')
        return adata[selected, :]

    return adata

def gradient_check(net,input,batch_size,output_dim,input_dim):
    net.zero_grad()

    if len(input)==1:
        output = net.encode(input)

        g = torch.zeros(batch_size, output_dim, input_dim)

        for i in range(output_dim):
            g[:, i] = torch.autograd.grad(output[:, i].sum(), input, retain_graph=True)[0].data


        return g
    return 0