
import argparse
import copy
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pandas.core.arrays import boolean
from scipy import stats
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (auc, average_precision_score,
                             classification_report, mean_squared_error,
                             precision_recall_curve, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import dropout, layer_norm, nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import gae.utils as gut
import graph_function as g
import models
import sampling as sam
import utils as ut
import trainers as t
from gae.utils import get_roc_score, mask_test_edges, preprocess_graph
from models import (AEBase, GAEBase, GCNPredictor, Predictor,
                    PretrainedPredictor, PretrainedVAEPredictor, VAEBase)
import matplotlib

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
    select_drug = args.drug
    na = args.missing_value
    data_path = args.data_path
    label_path = args.label_path
    test_size = args.test_size
    valid_size = args.valid_size
    g_disperson = args.var_genes_disp
    model_path = args.source_model_path
    encoder_path = args.encoder_path
    log_path = args.logging_file
    batch_size = args.batch_size
    encoder_hdims = args.encoder_h_dims.split(",")
    preditor_hdims = args.predictor_h_dims.split(",")
    reduce_model = args.dimreduce
    prediction = args.predition
    sampling = args.sampling

    encoder_hdims = list(map(int, encoder_hdims) )
    preditor_hdims = list(map(int, preditor_hdims) )
    load_model = bool(args.load_source_model)

    preditor_path = model_path + reduce_model + args.predictor+ prediction + select_drug + '.pkl'


    # Read data
    data_r=pd.read_csv(data_path,index_col=0)
    label_r=pd.read_csv(label_path,index_col=0)
    label_r=label_r.fillna(na)


    now=time.strftime("%Y-%m-%d-%H-%M-%S")

    log_path = log_path+now+".log"

    #log=open(log_path,"w")
    #sys.stdout=log

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


    # data = data_r

    # Filter out na values
    selected_idx = label_r.loc[:,select_drug]!=na

    if(g_disperson!=None):
        hvg,adata = ut.highly_variable_genes(data_r,min_disp=g_disperson)
        # Rename columns if duplication exist
        data_r.columns = adata.var_names
        # Extract hvgs
        data = data_r.loc[selected_idx,hvg]
    else:
        data = data_r.loc[selected_idx,:]

    # Extract labels
    label = label_r.loc[selected_idx,select_drug]

    # Scaling data
    mmscaler = preprocessing.MinMaxScaler()
    lbscaler = preprocessing.MinMaxScaler()

    data = mmscaler.fit_transform(data)
    label = label.values.reshape(-1,1)

    if prediction == "regression":
        label = lbscaler.fit_transform(label)
        dim_model_out = 1
    else:
        le = LabelEncoder()
        label = le.fit_transform(label)
        dim_model_out = 2

    #label = label.values.reshape(-1,1)

    logging.info(np.std(data))
    logging.info(np.mean(data))

    # Split traning valid test set
    X_train_all, X_test, Y_train_all, Y_test = train_test_split(data, label, test_size=test_size, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_all, Y_train_all, test_size=valid_size, random_state=42)
        # sampling method
    if sampling == None:
        X_train,Y_train=sam.nosampling(X_train,Y_train)
        logging.info("nosampling")
    elif sampling =="upsampling":
        X_train,Y_train=sam.upsampling(X_train,Y_train)
        logging.info("upsampling")
    elif sampling =="downsampling":
        X_train,Y_train=sam.downsampling(X_train,Y_train)
        logging.info("downsampling")
    elif  sampling=="SMOTE":
        X_train,Y_train=sam.SMOTEsampling(X_train,Y_train)
        logging.info("SMOTE")
    else:
        logging.info("not a legal sampling method")
    
    logging.info(data.shape)
    logging.info(label.shape)
    #logging.info(X_train.shape, Y_train.shape)
    #logging.info(X_test.shape, Y_test.shape)
    logging.info(X_train.max())
    logging.info(X_train.min())

    # Select the Training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    logging.info(device)
    torch.cuda.set_device(device)

    # Construct datasets and data loaders
    X_trainTensor = torch.FloatTensor(X_train).to(device)
    X_validTensor = torch.FloatTensor(X_valid).to(device)
    X_testTensor = torch.FloatTensor(X_test).to(device)
    X_allTensor = torch.FloatTensor(data).to(device)

    if prediction  == "regression":
        Y_trainTensor = torch.FloatTensor(Y_train).to(device)
        Y_trainallTensor = torch.FloatTensor(Y_train_all).to(device)
        Y_validTensor = torch.FloatTensor(Y_valid).to(device)
    else:
        Y_trainTensor = torch.LongTensor(Y_train).to(device)
        Y_trainallTensor = torch.LongTensor(Y_train_all).to(device)
        Y_validTensor = torch.LongTensor(Y_valid).to(device)


    train_dataset = TensorDataset(X_trainTensor, X_trainTensor)
    valid_dataset = TensorDataset(X_validTensor, X_validTensor)
    test_dataset = TensorDataset(X_testTensor, X_testTensor)
    all_dataset = TensorDataset(X_allTensor, X_allTensor)

    X_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    X_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    X_allDataLoader = DataLoader(dataset=all_dataset, batch_size=batch_size, shuffle=True)

    # construct TensorDataset
    trainreducedDataset = TensorDataset(X_trainTensor, Y_trainTensor)
    validreducedDataset = TensorDataset(X_validTensor, Y_validTensor)

    trainDataLoader_p = DataLoader(dataset=trainreducedDataset, batch_size=batch_size, shuffle=True)
    validDataLoader_p = DataLoader(dataset=validreducedDataset, batch_size=batch_size, shuffle=True)

    dataloaders_train = {'train':trainDataLoader_p,'val':validDataLoader_p}

    if(bool(args.pretrain)!=False):
        dataloaders_pretrain = {'train':X_trainDataLoader,'val':X_validDataLoader}
        if reduce_model == "VAE":
            encoder = VAEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
        else:
            encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
        
        if torch.cuda.is_available():
            encoder.cuda()

        logging.info(encoder)
        encoder.to(device)

        optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
        loss_function_e = nn.MSELoss()
        exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

        if reduce_model == "AE":
            encoder,loss_report_en = t.train_AE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                        optimizer=optimizer_e,loss_function=loss_function_e,
                                        n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=encoder_path)
        elif reduce_model == "VAE":
            encoder,loss_report_en = t.train_VAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                            optimizer=optimizer_e,
                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=encoder_path)
        
        logging.info("Pretrained finished")

    # Train model of predictor 

    if args.predictor == "DNN":
        if reduce_model == "AE":
            model = PretrainedPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                                    hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                                    pretrained_weights=encoder_path,freezed=bool(args.freeze_pretrain))
        elif reduce_model == "VAE":
            model = PretrainedVAEPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                            hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                            pretrained_weights=encoder_path,freezed=bool(args.freeze_pretrain),z_reparam=bool(args.VAErepram))
           
    elif args.predictor == "GCN":

        if reduce_model == "VAE":
            gcn_encoder = VAEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)
        else:
            gcn_encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims)

        gcn_encoder.load_state_dict(torch.load(args.GCNreduce_path))    
        gcn_encoder.to(device)

        train_embeddings = gcn_encoder.encode(X_trainTensor)
        zOut_tr = train_embeddings.cpu().detach().numpy()
        valid_embeddings = gcn_encoder.encode(X_validTensor)
        zOut_va = valid_embeddings.cpu().detach().numpy()
        test_embeddings=gcn_encoder.encode(X_testTensor)
        zOut_te = test_embeddings.cpu().detach().numpy()

        adj_tr, edgeList_tr = g.generateAdj(zOut_tr, graphType='KNNgraphStatsSingleThread', para = 'euclidean'+':'+str('10'), adjTag =True)
        adj_va, edgeList_va = g.generateAdj(zOut_va, graphType='KNNgraphStatsSingleThread', para = 'euclidean'+':'+str('10'), adjTag =True)
        adj_te, edgeList_te = g.generateAdj(zOut_te, graphType='KNNgraphStatsSingleThread', para = 'euclidean'+':'+str('10'), adjTag =True)

        Adj_trainTensor = preprocess_graph(adj_tr)
        Adj_validTensor = preprocess_graph(adj_va)
        Adj_testTensor = preprocess_graph(adj_te)

        Z_trainTensor = torch.FloatTensor(zOut_tr).to(device)
        Z_validTensor = torch.FloatTensor(zOut_va).to(device)
        Z_testTensor = torch.FloatTensor(zOut_te).to(device)

        if(args.binarizied ==0):
            zDiscret_tr = zOut_tr>np.mean(zOut_tr,axis=0)
            zDiscret_tr = 1.0*zDiscret_tr
            zDiscret_va = zOut_va>np.mean(zOut_va,axis=0)
            zDiscret_va = 1.0*zDiscret_va
            zDiscret_te = zOut_te>np.mean(zOut_te,axis=0)
            zDiscret_te = 1.0*zDiscret_te

            Z_trainTensor = torch.FloatTensor(zDiscret_tr).to(device)
            Z_validTensor = torch.FloatTensor(zDiscret_va).to(device)
            Z_testTensor = torch.FloatTensor(zDiscret_te).to(device)

        ZTensors_train = {'train':Z_trainTensor,'val':Z_validTensor}
        XTensors_train = {'train':X_trainTensor,'val':X_validTensor}

        YTensors_train = {'train':Y_trainTensor,'val':Y_validTensor}
        AdjTensors_train = {'train':Adj_trainTensor,'val':Adj_validTensor}

        if(args.GCNfeature=="x"):
            dim_GCNin = X_allTensor.shape[1]
            GCN_trainTensors = XTensors_train
            GCN_testTensor = X_testTensor
        else:
            dim_GCNin = Z_testTensor.shape[1]
            GCN_trainTensors = ZTensors_train
            GCN_testTensor = Z_testTensor

        model = GCNPredictor(input_feat_dim=dim_GCNin,hidden_dim1=encoder_hdims[0],hidden_dim2=dim_au_out, dropout=0.5,
                                hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                                pretrained_weights=encoder_path,freezed=bool(args.freeze_pretrain))

        # model2 = GAEBase(input_dim=X_train_all.shape[1], latent_dim=128,h_dims=[512])
        # model2.to(device)
        # test = model2((X_trainTensor,Adj_trainTensor))


    logging.info(model)
    if torch.cuda.is_available():
        model.cuda()
    model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    if prediction == "regression":
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.CrossEntropyLoss()

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    if args.predictor =="GCN":
        model,report = t.train_GCNpreditor_model(model=model,z=GCN_trainTensors,y=YTensors_train,adj=AdjTensors_train,
                                    optimizer=optimizer,loss_function=loss_function,n_epochs=epochs,scheduler=exp_lr_scheduler,save_path=preditor_path)

    else:
        model,report = t.train_predictor_model(model,dataloaders_train,
                                            optimizer,loss_function,epochs,exp_lr_scheduler,load=load_model,save_path=preditor_path)
    if args.predictor != 'GCN':
        dl_result = model(X_testTensor).detach().cpu().numpy()
    else:
        dl_result = model(GCN_testTensor,Adj_testTensor).detach().cpu().numpy()

    #torch.save(model.feature_extractor.state_dict(), preditor_path+"encoder.pkl")


    logging.info('Performances: R/Pearson/Mse/')

    if prediction == "regression":
        logging.info(r2_score(dl_result,Y_test))
        logging.info(pearsonr(dl_result.flatten(),Y_test.flatten()))
        logging.info(mean_squared_error(dl_result,Y_test))
    else:
        lb_results = np.argmax(dl_result,axis=1)
        #pb_results = np.max(dl_result,axis=1)
        pb_results = dl_result[:,1]

        report_dict = classification_report(Y_test, lb_results, output_dict=True)
        report_df = pd.DataFrame(report_dict).T
        ap_score = average_precision_score(Y_test, pb_results)
        auroc_score = roc_auc_score(Y_test, pb_results)

        report_df['auroc_score'] = auroc_score
        report_df['ap_score'] = ap_score

        report_df.to_csv("saved/logs/" + reduce_model + args.predictor+ prediction + select_drug+now + '_report.csv')

        logging.info(classification_report(Y_test, lb_results))
        logging.info(average_precision_score(Y_test, pb_results))
        logging.info(roc_auc_score(Y_test, pb_results))

        model = DummyClassifier(strategy='stratified')
        model.fit(X_train, Y_train)
        yhat = model.predict_proba(X_test)
        naive_probs = yhat[:, 1]

        ut.plot_roc_curve(Y_test, naive_probs, pb_results, title=str(roc_auc_score(Y_test, pb_results)),
                            path="saved/figures/" + reduce_model + args.predictor+ prediction + select_drug+now + '_roc.pdf')
        ut.plot_pr_curve(Y_test,pb_results,  title=average_precision_score(Y_test, pb_results),
                        path="saved/figures/" + reduce_model + args.predictor+ prediction + select_drug+now + '_prc.pdf')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--data_path', type=str, default='data/GDSC2_expression.csv')
    parser.add_argument('--label_path', type=str, default='data/GDSC2_label_9drugs_binary.csv')
    parser.add_argument('--result_path', type=str, default='saved/logs/result_')
    parser.add_argument('--drug', type=str, default='Cisplatin')
    parser.add_argument('--missing_value', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--var_genes_disp', type=float, default=None)
    parser.add_argument('--sampling', type=str, default=None)

    # trainv
    parser.add_argument('--encoder_path','-e', type=str, default='saved/models/encoder_ae.pkl')
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--dimreduce', type=str, default="AE")
    parser.add_argument('--predictor', type=str, default="DNN")
    parser.add_argument('--predition', type=str, default="classification")
    parser.add_argument('--freeze_pretrain', type=int, default=1)
    parser.add_argument('--encoder_h_dims', type=str, default="2048,1024")
    parser.add_argument('--predictor_h_dims', type=str, default="256,128")
    parser.add_argument('--GCNreduce_path', type=str, default='saved/models/encoder_ae.pkl')
    parser.add_argument('--binarizied', type=int, default=0)
    parser.add_argument('--GCNfeature', type=str, default="z")
    parser.add_argument('--VAErepram', type=int, default=0)



    # misc
    parser.add_argument('--message', '-m',  type=str, default='')
    parser.add_argument('--output_name', '-n',  type=str, default='')
    parser.add_argument('--source_model_path', '-p',  type=str, default='saved/models/source_model_')
    parser.add_argument('--logging_file', '-l',  type=str, default='saved/logs/log')
    parser.add_argument('--load_source_model',  type=int, default=1)
    warnings.filterwarnings("ignore")

    args, unknown = parser.parse_known_args()
    matplotlib.use('Agg')

    run_main(args)

