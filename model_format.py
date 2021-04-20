#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import sys
import time
from decimal import Decimal

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from captum.attr import IntegratedGradients
from numpy.lib.function_base import gradient
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics import (auc, average_precision_score,
                             classification_report, mean_squared_error,
                             precision_recall_curve, r2_score, roc_auc_score)
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
import DaNN.mmd as mmd
import scanpypip.preprocessing as pp
import trainers as t
import utils as ut
from models import (AEBase, CVAEBase, DaNN, Predictor, PretrainedPredictor,
                    PretrainedVAEPredictor, TargetModel, VAEBase)
from scanpypip.utils import get_de_dataframe
from trajectory import trajectory

DATA_MAP={
"GSE117872":"data/GSE117872/GSE117872_good_Data_TPM.txt",
"GSE117309":'data/GSE117309/filtered_gene_bc_matrices_HBCx-22/hg19/',
"GSE117309_TAMR":'data/GSE117309/filtered_gene_bc_matrices_HBCx22-TAMR/hg19/',
"GSE121107":'data/GSE121107/GSM3426289_untreated_out_gene_exon_tagged.dge.txt',
"GSE121107_1H":'data/GSE121107/GSM3426290_entinostat_1hr_out_gene_exon_tagged.dge.txt',
"GSE121107_6H":'data/GSE121107/GSM3426291_entinostat_6hr_out_gene_exon_tagged.dge.txt',
"GSE111014":'data/GSE111014/',
"GSE110894":"data/GSE110894/GSE110894.csv",
"GSE122843":"data/GSE122843/GSE122843.txt",
"GSE112274":"data/GSE112274/GSE112274_cell_gene_FPKM.csv",
"GSE116237":"data/GSE116237/GSE116237_bulkRNAseq_expressionMatrix.txt",
"GSE108383":"data/GSE108383/GSE108383_Melanoma_fluidigm.txt",
"GSE140440":"data/GSE140440/GSE140440.csv",
"GSE129730":"data/GSE129730/GSE129730.h5ad",
"GSE149383":"../data/GSE149383/erl_total_data_2K.csv"

}

REMOVE_GENES=["mt-","rps","rpl"]

def run_main(args):
################################################# START SECTION OF LOADING PARAMETERS #################################################
    # Read parameters
    epochs = args.epochs
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    freeze = args.freeze_pretrain
    source_model_path = args.source_model_path
    target_model_path = args.target_model_path 
    log_path = args.logging_file
    encoder_hdims = args.source_h_dims.split(",")
    encoder_hdims = list(map(int, encoder_hdims))
    pretrain = args.pretrain
    prediction = args.predition
    reduce_model = args.dimreduce
    predict_hdims = args.p_h_dims.split(",")
    predict_hdims = list(map(int, predict_hdims))
    load_model = True

    
    # Misc
    now=time.strftime("%Y-%m-%d-%H-%M-%S")
    # Initialize logging and std out
    out_path = log_path+now+".err"
    log_path = log_path+now+".log"

    out=open(out_path,"w")
    sys.stderr=out
    
    #Logging infomaion
    logging.basicConfig(level=logging.INFO,
                    filename=log_path,
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    logging.getLogger('matplotlib.font_manager').disabled = True


    logging.info(args)
    
    # Save arguments
    args_df = ut.save_arguments(args,now)
################################################# END SECTION OF LOADING PARAMETERS #################################################

    # Select the device of gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    logging.info(device)
    torch.cuda.set_device(device)
################################################# END SECTION OF LOADING BULK DATA  #################################################

################################################# START SECTION OF MODEL CUNSTRUCTION  #################################################
    # Construct target encoder
    if reduce_model == "AE":
        encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
        loss_function_e = nn.MSELoss()
    elif reduce_model == "VAE":
        encoder = VAEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
    # elif reduce_model == "CVAE":
    #     # Number of condition is equal to the number of clusters
    #     encoder = CVAEBase(input_dim=data.shape[1],n_conditions=len(set(data_c)),latent_dim=dim_au_out,h_dims=encoder_hdims)

    if torch.cuda.is_available():
        encoder.cuda()

    logging.info("Target encoder structure is: ")
    logging.info(encoder)

    encoder.to(device)
    optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
    loss_function_e = nn.MSELoss()
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)


    # Load source model before transfer
    if prediction == "regression":
            dim_model_out = 1
    else:
            dim_model_out = 2
    # Load AE model
    if reduce_model == "AE":

        source_model = PretrainedPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                pretrained_weights=None,freezed=freeze)
        source_model.load_state_dict(torch.load(source_model_path))
        source_encoder = source_model
    # Load VAE model
    elif reduce_model in ["VAE","CVAE"]:
        source_model = PretrainedVAEPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                pretrained_weights=None,freezed=freeze,z_reparam=bool(args.VAErepram))
        source_model.load_state_dict(torch.load(source_model_path))
        source_encoder = source_model
    logging.info("Load pretrained source model from: "+source_model_path)
           
    source_encoder.to(device)
################################################# END SECTION OF MODEL CUNSTRUCTION  #################################################

################################################# START SECTION OF SC MODEL PRETRAININIG  #################################################
    # Pretrain target encoder
    # Pretain using autoencoder is pretrain is not False
    if(str(pretrain)!='0'):
        # Pretrained target encoder if there are not stored files in the harddisk
        train_flag = True
        pretrain = str(pretrain)
        
        try:
            encoder.load_state_dict(torch.load(pretrain))
            logging.info("Load pretrained target encoder from "+pretrain)
            train_flag = False

        except:
            logging.warning("Loading failed, procceed to re-train model")


            logging.info("Pretrained finished")
################################################# END SECTION OF SC MODEL PRETRAININIG  #################################################

################################################# START SECTION OF TRANSFER LEARNING TRAINING #################################################
    # Using ADDA transfer learning
    if args.transfer =='ADDA':

        # Set discriminator model
        discriminator = Predictor(input_dim=dim_au_out,output_dim=2)
        discriminator.to(device)
        loss_d = nn.CrossEntropyLoss()
        optimizer_d = optim.Adam(encoder.parameters(), lr=1e-2)
        exp_lr_scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d)

        # Adversairal trainning
        discriminator,encoder, report_, report2_ = t.train_ADDA_model(source_encoder,encoder,discriminator,
                            dataloaders_source,dataloaders_pretrain,
                            loss_d,loss_d,
                            # Should here be all optimizer d?
                            optimizer_d,optimizer_d,
                            exp_lr_scheduler_d,exp_lr_scheduler_d,
                            epochs,device,
                            target_model_path)

        logging.info("Transfer ADDA finished")
        

    # DaNN model
    elif args.transfer == 'DaNN':

        # Set predictor loss
        loss_d = nn.CrossEntropyLoss()
        optimizer_d = optim.Adam(encoder.parameters(), lr=1e-2)
        exp_lr_scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d)

        # Set DaNN model
        DaNN_model = DaNN(source_model=source_encoder,target_model=encoder)
        DaNN_model.to(device)

        def loss(x,y,GAMMA=args.GAMMA_mmd):
            result = mmd.mmd_loss(x,y,GAMMA)
            return result

        loss_disrtibution = loss

        # Tran DaNN model
        DaNN_model, report_ = t.train_DaNN_model(DaNN_model,
                            dataloaders_source,dataloaders_pretrain,
                            # Should here be all optimizer d?
                            optimizer_d, loss_d,
                            epochs,exp_lr_scheduler_d,
                            dist_loss=loss_disrtibution,
                            load=load_model,
                            weight = args.mmd_weight,
                            save_path=target_model_path+"_DaNN.pkl")

        encoder = DaNN_model.target_model
        source_model = DaNN_model.source_model
        logging.info("Transfer DaNN finished")

################################################# END SECTION OF TRANSER LEARNING TRAINING #################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--mmd_weight', type=float, default=0.25)

    # train
    parser.add_argument('--source_model_path','-s', type=str, default='saved/models/BET_dw_512_AE.pkl')
    parser.add_argument('--target_model_path', '-p',  type=str, default='saved/models/GSE110894_I-BET-762512AE')
    parser.add_argument('--pretrain', type=str, default='saved/models/GSE110894_I-BET-762_512_ae.pkl')
    parser.add_argument('--transfer', type=str, default="DaNN")

    parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--dimreduce', type=str, default="AE")
    parser.add_argument('--predictor', type=str, default="DNN")
    parser.add_argument('--freeze_pretrain', type=int, default=0)
    parser.add_argument('--source_h_dims', type=str, default="2048,1024")
    parser.add_argument('--target_h_dims', type=str, default="2048,1024")
    parser.add_argument('--p_h_dims', type=str, default="128,64")
    parser.add_argument('--predition', type=str, default="classification")
    parser.add_argument('--VAErepram', type=int, default=1)

    parser.add_argument('--message', '-m',  type=str, default='message')
    parser.add_argument('--output_name', '-n',  type=str, default='saved/results')
    parser.add_argument('--logging_file', '-l',  type=str, default='saved/logs/transfer_')

    #
    args, unknown = parser.parse_known_args()
    run_main(args)
