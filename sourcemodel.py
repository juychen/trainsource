
import argparse
import copy
import os

import numpy as np
import pandas as pd
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--data_path', type=str, default='data/GDSC2_expression.csv')
    parser.add_argument('--label_path', type=str, default='data/GDSC2_label_9drugs.csv')
    parser.add_argument('--drug', type=str, default='Tamoxifen')
    parser.add_argument('--missing_value', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--vairable_genes_dispersion', type=float, default=0)

    # train
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--dimreduce', type=str, default="AE")
    parser.add_argument('--predictor', type=str, default="DNN")

    # misc
    parser.add_argument('--message', '-m',  type=str, default='')
    parser.add_argument('--output_name', '-n',  type=str, default='')

    args, unknown = parser.parse_known_args()
    main(args)


def main(args):

    # Define parameters
    epochs = args.epochs
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    dim_dnn_in = dim_au_out
    dim_dnn_out=1

    data_path = args.data_path
    label_path = args.label_path

    data_r=pd.read_csv(data_path)
    label_r=pd.read_csv(label_path)

    label_r=label_r.fillna(1)
    data = data_r
    label = label_r.iloc[:,10]
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    data = scaler.fit_transform(data)

    print(np.std(data))
    print(np.mean(data))

    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, random_state=42)

    print(data.shape)
    print(label.shape)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    print(X_train.max())
    print(X_train.min())


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    torch.cuda.set_device(device)


    trainData = torch.FloatTensor(X_train).to(device)
    testData = torch.FloatTensor(X_test).to(device)
    y = torch.FloatTensor(Y_train.values).to(device)
    allData = torch.FloatTensor(data).to(device)

    # construct TensorDataset
    train_dataset = TensorDataset(trainData, trainData)
    test_dataset = TensorDataset(testData, testData)
    all_dataset = TensorDataset(allData, allData)

    trainDataLoader1 = DataLoader(dataset=train_dataset, batch_size=200, shuffle=False)
    trainDataLoaderall = DataLoader(dataset=all_dataset, batch_size=200, shuffle=False)


    autoencoder = AE(dim_au_in = X_train.shape[1],dim_au_out=dim_au_out).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    loss_func = nn.SmoothL1Loss().to(device)
    loss_train = np.zeros((epochs, 1))



    for epoch in range(epochs):
        # 
        for batchidx, (x, _) in enumerate(trainDataLoaderall):
            x.requires_grad_(True)
            # encode and decode 
            decoded, encoded = autoencoder(x)
            # compute loss
            print(encoded.shape, decoded.shape)
            loss = loss_func(decoded, x)      
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_train[epoch,0] = loss.item()  
        print('Epoch: %04d, Training loss=%.8f' %
            (epoch+1, loss.item()))
    
    torch.save(autoencoder.state_dict(), 'saved/models/'+data_path+args.dimreduce+'.pkl')
    
    # extract features
    _, encodedTrainData = autoencoder(trainData)
    featureTensor = encodedTrainData.double()
    feature = featureTensor.detach().cpu().numpy()



    print(feature.shape)


    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(feature, Y_train.values)


    # In[23]:


    _,testFeature = autoencoder(testData)
    lasso = clf.predict(testFeature.detach().cpu().numpy())


    print(r2_score(lasso,Y_test))

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(feature)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], label="label")
    plt.legend()
    plt.savefig("saved/figrues/"+data_path+args.dimreduce+args.predictor+".png")


    EPOCH = 500


    # In[30]:


    # Load data
    # data type conversion
    B_feature = torch.FloatTensor(feature).to(device)
    y = torch.FloatTensor(Y_train.values).to(device)
    # construct TensorDataset
    b_data = TensorDataset(B_feature, y)
    trainDataLoader2 = DataLoader(dataset=b_data, batch_size=200, shuffle=False)


    # In[31]:



    predictor = DNN(dim_dnn_in, dim_dnn_out).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=1e-3,betas=(0.9,0.99))
    loss_func = nn.BCELoss().to(device)
    loss_train = np.zeros((epochs, 1))

    # train model
    for epoch in range(EPOCH):
        print('Epoch: ',epoch)
        for step,(batch_x,batch_y) in enumerate(trainDataLoader2):
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)
            # predict label
            output = predictor(b_x)
            # b_y=F.sigmoid(b_y) 
            
            #print
            #print(output)
            #print(b_y)
            # compute loss
            loss = loss_func(output,b_y)
            #loss = criterion(output, b_y)
            
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_train[epoch,0] = loss.item()  
        print('Epoch: %04d, Training loss=%.8f' %
            (epoch+1, loss.item())) 

    torch.save(predictor.state_dict(), 'saved/models/'+data_path+args.predictor+'.pkl')


    # Get tesing feature
    _,testFeature = autoencoder(testData)
    testpredict = predictor(testFeature)
    r2_score(testpredict.detach().cpu().numpy(),Y_test)
    mean_squared_error(testpredict.detach().cpu().numpy(),Y_test)
