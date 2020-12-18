#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time
import logging
import sys

import numpy as np
from numpy.lib.function_base import gradient
import pandas as pd
import scanpy as sc
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import scanpypip.preprocessing as pp
import utils as ut
import trainers as t
from models import AEBase, DaNN, Predictor, PretrainedPredictor,VAEBase,PretrainedVAEPredictor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (auc, average_precision_score,
                             classification_report, mean_squared_error,
                             precision_recall_curve, r2_score, roc_auc_score)
from decimal import Decimal
from sklearn.metrics.cluster import adjusted_rand_score


DATA_MAP={
"GSE117872":"data/GSE117872/GSE117872_good_Data_TPM.txt",
"GSE117309":'data/GSE117309/filtered_gene_bc_matrices_HBCx-22/hg19/',
"GSE117309_TAMR":'data/GSE117309/filtered_gene_bc_matrices_HBCx22-TAMR/hg19/',
"GSE121107":'data/GSE121107/GSM3426289_untreated_out_gene_exon_tagged.dge.txt',
"GSE121107_1H":'data/GSE121107/GSM3426290_entinostat_1hr_out_gene_exon_tagged.dge.txt',
"GSE121107_6H":'data/GSE121107/GSM3426291_entinostat_6hr_out_gene_exon_tagged.dge.txt',
"GSE111014":'data/GSE111014/'
}

def run_main(args):

    epochs = args.epochs
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    na = args.missing_value
    data_path = DATA_MAP[args.target_data]
    test_size = args.test_size
    select_drug = args.drug

    freeze = args.freeze_pretrain

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
    prediction = args.predition
    data_name = args.target_data
    label_path = args.label_path

    reduce_model = args.dimreduce
    predict_hdims = args.p_h_dims.split(",")
    predict_hdims = list(map(int, predict_hdims))
    leiden_res = args.cluster_res

    load_model = bool(args.load_target_model)

    
    # Misc
    now=time.strftime("%Y-%m-%d-%H-%M-%S")

    # Initialize logging and std out
    out_path = log_path+now+".err"
    log_path = log_path+now+".log"

    out=open(out_path,"w")
    sys.stderr=out
    
    #Logging infomaion
    logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename=log_path,
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )
    logging.getLogger('matplotlib.font_manager').disabled = True


    logging.info(args)
    # Save arguments
    args_df = ut.save_arguments(args,now)

    #os.mkdir('figures/'+now)

    # Load data and preprocessing
    adata = pp.read_sc_file(data_path)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    adata = pp.cal_ncount_ngenes(adata)

    # Show statisctic after QX
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                jitter=0.4, multi_panel=True,save=data_name,show=False)
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt',show=False)
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts',show=False)

    #Preprocess data by filtering
    adata = pp.receipe_my(adata,l_n_genes=min_n_genes,r_n_genes=max_n_genes,filter_mincells=args.min_c,
                        filter_mingenes=args.min_g,normalize=True,log=True)


    # Select highly variable genes
    sc.pp.highly_variable_genes(adata,min_disp=g_disperson,max_disp=np.inf,max_mean=6)
    sc.pl.highly_variable_genes(adata,save=data_name,show=False)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]

    # Preprocess data if spcific process is required
    if data_name == 'GSE117872':
        adata =  ut.specific_process(adata,dataname=data_name,select_origin=args.batch_id)
        data=adata.X

    else:
        data=adata.X
 

    #Prepare to normailize and split target data
    mmscaler = preprocessing.MinMaxScaler()

    try:
        data = mmscaler.fit_transform(data)

    except:
        logging.warning("Only one class, no ROC")

        # Process sparse data
        data = data.todense()
        data = mmscaler.fit_transform(data)

    # Split data to train and valid set
    Xtarget_train, Xtarget_valid = train_test_split(data, test_size=valid_size, random_state=42)


    # Select the device of gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    logging.info(device)
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
    if reduce_model == "AE":
        encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
        loss_function_e = nn.MSELoss()
    else:
        encoder = VAEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)

    if torch.cuda.is_available():
        encoder.cuda()

    logging.info("Target encoder structure is: ")
    logging.info(encoder)

    encoder.to(device)
    optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
    loss_function_e = nn.MSELoss()
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)


    # Read source data
    data_r=pd.read_csv(source_data_path,index_col=0)
    label_r=pd.read_csv(label_path,index_col=0)
    label_r=label_r.fillna(na)

    # Extract labels
    selected_idx = label_r.loc[:,select_drug]!=na
    label = label_r.loc[selected_idx,select_drug]

    label = label.values.reshape(-1,1)

    if prediction == "regression":
        lbscaler = preprocessing.MinMaxScaler()
        label = lbscaler.fit_transform(label)
        dim_model_out = 1
    else:
        le = preprocessing.LabelEncoder()
        label = le.fit_transform(label)
        dim_model_out = 2

    # Process source data
    mmscaler = preprocessing.MinMaxScaler()
    source_data = mmscaler.fit_transform(data_r)

    # Split source data
    Xsource_train_all, Xsource_test, Ysource_train_all, Ysource_test = train_test_split(source_data,label, test_size=test_size, random_state=42)
    Xsource_train, Xsource_valid, Ysource_train, Ysource_valid = train_test_split(Xsource_train_all,Ysource_train_all, test_size=valid_size, random_state=42)

    # Transform source data
    # Construct datasets and data loaders
    Xsource_trainTensor = torch.FloatTensor(Xsource_train).to(device)
    Xsource_validTensor = torch.FloatTensor(Xsource_valid).to(device)
    #Xsource_testTensor = torch.FloatTensor(Xsource_test).to(device)
    #Xsource_allTensor = torch.FloatTensor(source_data).to(device)

    if prediction  == "regression":
        Ysource_trainTensor = torch.FloatTensor(Ysource_train).to(device)
        #Ysource_trainallTensor = torch.FloatTensor(Ysource_train_all).to(device)
        Ysource_validTensor = torch.FloatTensor(Ysource_valid).to(device)
    else:
        Ysource_trainTensor = torch.LongTensor(Ysource_train).to(device)
        #Ysource_trainallTensor = torch.LongTensor(Ysource_train_all).to(device)
        Ysource_validTensor = torch.LongTensor(Ysource_valid).to(device)


    sourcetrain_dataset = TensorDataset(Xsource_trainTensor, Ysource_trainTensor)
    sourcevalid_dataset = TensorDataset(Xsource_validTensor, Ysource_validTensor)
    #sourcetest_dataset = TensorDataset(Xsource_testTensor, Ysource_trainallTensor)
    #sourceall_dataset = TensorDataset(Xsource_allTensor, Ysource_validTensor)

    Xsource_trainDataLoader = DataLoader(dataset=sourcetrain_dataset, batch_size=batch_size, shuffle=True)
    Xsource_validDataLoader = DataLoader(dataset=sourcevalid_dataset, batch_size=batch_size, shuffle=True)
    #X_allDataLoader = DataLoader(dataset=sourceall_dataset, batch_size=batch_size, shuffle=True)



    dataloaders_source = {'train':Xsource_trainDataLoader,'val':Xsource_validDataLoader}


    # Load source model

    if prediction == "regression":
            dim_model_out = 1
    else:
            dim_model_out = 2

    if reduce_model == "AE":


        source_model = PretrainedPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                pretrained_weights=None,freezed=freeze)
        source_model.load_state_dict(torch.load(source_model_path))

        source_encoder = source_model
    else:
        # source_encoder = VAEBase(input_dim=data_r.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
        # source_encoder.load_state_dict(torch.load(source_model_path))

        source_model = PretrainedVAEPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                pretrained_weights=None,freezed=freeze,z_reparam=bool(args.VAErepram))
        source_model.load_state_dict(torch.load(source_model_path))

        source_encoder = source_model
    logging.info("Load pretrained source model from: "+source_model_path)


           
    source_encoder.to(device)

    # Pretrain target encoder
    if(str(pretrain)!='0'):
        if(os.path.exists(pretrain)==False):
            pretrain = str(pretrain)

            if reduce_model == "AE":
                encoder,loss_report_en = t.train_AE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                            optimizer=optimizer_e,loss_function=loss_function_e,
                                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain)
                logging.info("Pretrained finished")
            else:
                encoder,loss_report_en = t.train_VAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                optimizer=optimizer_e,
                                n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain)

        else:
            pretrain = str(pretrain)
            encoder.load_state_dict(torch.load(pretrain))
            logging.info("Load pretrained target encoder from "+pretrain)


        # Extract pretrain feature
        embeddings_p = encoder.encode(X_allTensor).detach().cpu().numpy()
        # Add embeddings to the adata package
        adata.obsm["X_pre"] = embeddings_p


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

    elif args.transfer == 'DaNN':

        # Set discriminator model
        loss_d = nn.CrossEntropyLoss()
        optimizer_d = optim.Adam(encoder.parameters(), lr=1e-2)
        exp_lr_scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d)

        # Adversairal trainning
        DaNN_model = DaNN(source_model=source_encoder,target_model=encoder)
        DaNN_model.to(device)

        DaNN_model, report_ = t.train_DaNN_model(DaNN_model,
                            dataloaders_source,dataloaders_pretrain,
                            # Should here be all optimizer d?
                            optimizer_d, loss_d,
                            epochs,exp_lr_scheduler_d,
                            load=load_model,
                            save_path=target_model_path+"_DaNN.pkl")

        encoder = DaNN_model.target_model
        source_model = DaNN_model.source_model        
        logging.info("Transfer DaNN finished")



    # Extract feature
    # embeddings = encoder.encode(X_allTensor).detach().cpu().numpy()
    embedding_tensors = encoder.encode(X_allTensor)
    prediction_tensors = source_model.predictor(embedding_tensors)
    embeddings = embedding_tensors.detach().cpu().numpy()
    predictions = prediction_tensors.detach().cpu().numpy()

    # gradient = ut.gradient_check(net=encoder,input=Xtarget_trainTensor,batch_size=Xtarget_trainTensor.shape[0],
    #                                 output_dim=embeddings.shape[1],input_dim=Xtarget_trainTensor.shape[1])

    if(prediction=="regression"):
        adata.obs["sens_preds"] = predictions
    else:
        adata.obs["sens_preds"] = predictions[:,1]
        adata.obs["sens_label"] = predictions.argmax(axis=1)
        adata.obs["sens_label"] = adata.obs["sens_label"].astype('category')

 
    # PCA
    sc.tl.pca(adata, svd_solver='arpack')

    # Add embeddings to the adata package
    adata.obsm["X_Trans"] = embeddings

    # Generate neighbor graph
    sc.pp.neighbors(adata, n_neighbors=10,use_rep="X_Trans")
    #sc.tl.umap(adata)

    # Use t-sne 
    sc.tl.tsne(adata,use_rep="X_Trans")

    # Leiden on the data
    sc.tl.leiden(adata)

    # Plot tsne
    sc.pl.tsne(adata,save=data_name+now,color=["leiden"],show=False)

    # Differenrial expression genes
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False,save=data_name+now,show=False)


    title = "Cell scatter plot"
    if(data_name=='GSE117872'):

        label = adata.obs['cluster']
        if len(label[label != "Sensitive"] )>0:
            label[label != "Sensitive"] = 'Resistant'
        le_sc = LabelEncoder()
        le_sc.fit(['Resistant','Sensitive'])
        pb_results = adata.obs['sens_preds']
        Y_test = le_sc.transform(label)
        ap_score = average_precision_score(Y_test, pb_results)

        try:
            auroc_score = roc_auc_score(Y_test, pb_results)
        except:
            logging.warning("Only one class, no ROC")
            auroc_score = 0
        

        ap_title = "ap: "+str(Decimal(ap_score).quantize(Decimal('0.0000')))
        auroc_title = "roc: "+str(Decimal(auroc_score).quantize(Decimal('0.0000')))

        color_list = ["cluster","origin",'sens_preds']
        title_list = ['',ap_title,auroc_title]

        report_df = args_df
        report_df['auroc_score'] = auroc_score
        report_df['ap_score'] = ap_score

        report_df.to_csv("saved/logs/report" + reduce_model + args.predictor+ prediction + select_drug+now + '.csv')
    else:
        
        color_list = ["leiden",'sens_preds']
        title_list = ['',""]
    # Simple analysis do neighbors in adata
    # Plot umap
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata,resolution=leiden_res)
    sc.pl.umap(adata,color=color_list,save=data_name+"_umap_"+now,show=False,title=title_list)
    # Plot umap
    sc.pp.neighbors(adata,use_rep='X_Trans',key_added="Trans")
    sc.tl.umap(adata,neighbors_key="Trans")
    sc.tl.leiden(adata,neighbors_key="Trans",key_added="leiden_trans",resolution=leiden_res)
    sc.pl.umap(adata,color=color_list,neighbors_key="Trans",save=data_name+"_umap_TL"+now,show=False,title=title_list)
    # Plot tsne
    sc.pl.tsne(adata,color=color_list,neighbors_key="Trans",save=data_name+"_tsne-TL_"+now,show=False,title=title_list)
    # Plot tsne pretrained
    sc.pp.neighbors(adata,use_rep='X_pre',key_added="Pret")
    sc.tl.umap(adata,neighbors_key="Pret")
    sc.tl.leiden(adata,neighbors_key="Pret",key_added="leiden_Pret",resolution=leiden_res)
    sc.pl.umap(adata,color=["leiden_trans"],neighbors_key="Pret",save=data_name+"_tsne_Pretrain_"+now,show=False)

    if(data_name!='GSE117872'):
        ari_score_trans  = adjusted_rand_score(adata.obs['leiden_trans'],adata.obs['sens_label'])
        ari_score = adjusted_rand_score(adata.obs['leiden'],adata.obs['sens_label'])

        report_df = args_df
        report_df['ari_score'] = ari_score
        report_df['ari_trans_score'] = ari_score_trans
        report_df.to_csv("saved/logs/report" + reduce_model + args.predictor+ prediction + select_drug+now + '.csv')


    # Save adata
    adata.write("saved/adata/"+data_name+now+".h5ad")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--source_data', type=str, default='data/GDSC2_expression.csv')
    parser.add_argument('--label_path', type=str, default='data/GDSC2_label_9drugs_binary.csv')
    parser.add_argument('--target_data', type=str, default="GSE117872")
    parser.add_argument('--drug', type=str, default='Cisplatin')
    parser.add_argument('--missing_value', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--var_genes_disp', type=float, default=0)
    parser.add_argument('--min_n_genes', type=int, default=0)
    parser.add_argument('--max_n_genes', type=int, default=20000)
    parser.add_argument('--min_g', type=int, default=200)
    parser.add_argument('--min_c', type=int, default=3)
    parser.add_argument('--cluster_res', type=int, default=0.3)


    # train
    parser.add_argument('--source_model_path','-s', type=str, default='saved/models/source_model_VAEDNNclassificationCisplatin.pkl')
    parser.add_argument('--target_model_path', '-p',  type=str, default='saved/models/transfer_')
    parser.add_argument('--pretrain', type=str, default='saved/models/target_encoder_vae.pkl')
    parser.add_argument('--transfer', type=str, default="DaNN")

    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--dimreduce', type=str, default="VAE")
    parser.add_argument('--predictor', type=str, default="DNN")
    parser.add_argument('--freeze_pretrain', type=int, default=1)
    parser.add_argument('--source_h_dims', type=str, default="2048,1024")
    parser.add_argument('--target_h_dims', type=str, default="512,256")
    parser.add_argument('--p_h_dims', type=str, default="256,128")
    parser.add_argument('--predition', type=str, default="classification")
    parser.add_argument('--VAErepram', type=int, default=1)
    parser.add_argument('--batch_id', type=str, default="all")
    parser.add_argument('--load_target_model', type=int, default=1)


    # misc
    parser.add_argument('--message', '-m',  type=str, default='message')
    parser.add_argument('--output_name', '-n',  type=str, default='saved/results')
    parser.add_argument('--logging_file', '-l',  type=str, default='saved/logs/transfer_')

    #
    args, unknown = parser.parse_known_args()
    run_main(args)
