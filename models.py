import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
#import scipy.io as sio
import numpy as np
import pandas as pd


# Model of AE
class AE(nn.Module):
    def __init__(self,dim_au_in,dim_au_out):
        super(AE, self).__init__()
        self.dim = dim_au_in
        self.fc1 = nn.Linear(dim_au_in, 1024)

        self.fc2 = nn.Linear(1024, dim_au_out)
        self.fc3 = nn.Linear(dim_au_out, 1024)

        self.fc4 = nn.Linear(1024, dim_au_in)            
                    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)
       
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        #return self.decode(z), x
        return self.decode(z), z
    
    def fit(self, x):
        return 


class DNN(nn.Module):
    def __init__(self,dim_dnn_in,dim_dnn_out):
        super(DNN, self).__init__()
        self.dim = dim_dnn_in
        #self.fc1 = nn.Linear(dim, 64)
        #self.fc2 = nn.Linear(64, 32)
        #self.fc3 = nn.Linear(32, 16)
        #self.fc4 = nn.Linear(16, num)
        self.fc1 = nn.Linear(dim_dnn_in, 128)
        self.fc2 = nn.Linear(128, 128)
        #self.fc3 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(128, dim_dnn_out)
        #self.fc4 = nn.Linear(64, num)
        #sigmoid=0/1
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        y_hat=self.sigmoid(self.fc3(x))    
        #return self.decode(z), x
        return y_hat
    

class VAE(nn.Module):
    def __init__(self,dim_au_in,dim_au_out=512):
        super(VAE, self).__init__()

        self.input_shape = dim_au_in
        self.bottleneck_shape = dim_au_out


        self.fc1 = nn.Linear(dim_au_in, 1024)
        self.fc21 = nn.Linear(1024, dim_au_out)
        self.fc22 = nn.Linear(1024, dim_au_out)
        self.fc3 = nn.Linear(dim_au_out, 1024)
        self.fc4 = nn.Linear(1024, dim_au_in)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
        BCE = reconstruction_function(recon_x, x)  # mse loss
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        return BCE + KLD