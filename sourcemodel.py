
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
    model_path = args.model_store_path
    pretrain_path = args.pretrain_path
    log_path = args.logging_file
    batch_size = args.batch_size

    # Read data
    data_r=pd.read_csv(data_path,index_col=0)
    label_r=pd.read_csv(label_path,index_col=0)
    label_r=label_r.fillna(na)


    now=time.strftime("%Y-%m-%d-%H-%M-%S")

    log_path = log_path+now+".txt"

    log=open(log_path,"w")
    sys.stdout=log


    data = data_r

    if(g_disperson!=0):
        hvg,adata = ut.highly_variable_genes(data_r,min_disp=g_disperson)
    
    # Select index
    selected_idx = label_r.loc[:,select_drug]!=na

    # Rename columns if duplication exist
    data_r.columns = adata.var_names

    # Extract hvgs
    if(g_disperson!=0):
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
    label = lbscaler.fit_transform(label)
    #label = label.values.reshape(-1,1)

    print(np.std(data))
    print(np.mean(data))

    # Split traning valid test set
    X_train_all, X_test, Y_train_all, Y_test = train_test_split(data, label, test_size=test_size, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_all, Y_train_all, test_size=valid_size, random_state=42)
    
    print(data.shape)
    print(label.shape)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    print(X_train.max())
    print(X_train.min())

    # Select the Training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    torch.cuda.set_device(device)

    # Construct datasets and data loaders
    X_trainTensor = torch.FloatTensor(X_train).to(device)
    X_validTensor = torch.FloatTensor(X_valid).to(device)
    X_testTensor = torch.FloatTensor(X_test).to(device)
    X_allTensor = torch.FloatTensor(data).to(device)

    Y_trainTensor = torch.FloatTensor(Y_train).to(device)
    Y_validTensor = torch.FloatTensor(Y_valid).to(device)

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
    validDataLoader_p = DataLoader(dataset=trainreducedDataset, batch_size=batch_size, shuffle=True)

    dataloaders_train = {'train':trainDataLoader_p,'val':validDataLoader_p}

    if(args.pretrain==True):
        encoder = AEBase(input_dim=data.shape[1],latent_dim=dim_au_out,hidden_dims=[2048,1024])
        #model = VAE(dim_au_in=data_r.shape[1],dim_au_out=128)
        if torch.cuda.is_available():
            encoder.cuda()
        encoder.to(device)
        optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
        loss_function_e = nn.MSELoss()
        exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)
        encoder,loss_report = ut.train_extractor_model(net=encoder,data_loaders=dataloaders_train,
                                    optimizer=optimizer_e,loss_function=loss_function_e,
                                    n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=pretrain_path)


    # Models 
    model = PretrainedPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,hidden_dims=[2048,1024], 
                            hidden_dims_predictor=[256,128],
                            pretrained_weights=pretrain_path,freezed=args.freeze_pretrain)
    
    print(model)
    if torch.cuda.is_available():
        model.cuda()
    model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_function = nn.MSELoss()
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    model,report = ut.train_predictor_model(model,dataloaders_train,
                                        optimizer,loss_function,epochs,exp_lr_scheduler,save_path=model_path)

    dl_result = model(X_testTensor).detach().cpu().numpy()
    print(r2_score(dl_result,Y_test))
    print(pearsonr(dl_result.flatten(),Y_test.flatten()))
    print(mean_squared_error(dl_result,Y_test))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--data_path', type=str, default='data/GDSC2_expression.csv')
    parser.add_argument('--label_path', type=str, default='data/GDSC2_label_9drugs.csv')
    parser.add_argument('--drug', type=str, default='Tamoxifen')
    parser.add_argument('--missing_value', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--var_genes_disp', type=float, default=0.5)

    # train
    parser.add_argument('--pretrain_path', type=str, default='saved/models/pretrained.pkl')
    parser.add_argument('--pretrain', type=boolean, default=False)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--dimreduce', type=str, default="AE")
    parser.add_argument('--predictor', type=str, default="DNN")
    parser.add_argument('--freeze_pretrain', type=boolean, default=False)


    # misc
    parser.add_argument('--message', '-m',  type=str, default='')
    parser.add_argument('--output_name', '-n',  type=str, default='')
    parser.add_argument('--model_store_path', '-p',  type=str, default='saved/models/model.pkl')
    parser.add_argument('--logging_file', '-l',  type=str, default='saved/logs/log')

    #
    args, unknown = parser.parse_known_args()
    run_main(args)

