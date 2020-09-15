import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch import Tensor
from torch.nn import Dropout
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


# Model of AE
class AEBase(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 hidden_dims=[512],
                 drop_out=0.3):
                 
        super(AEBase, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        
        hidden_dims.insert(0,input_dim)

        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(drop_out))
            )
            #in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.bottleneck = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                                       hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                    nn.Dropout(drop_out))
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-2],
                                       hidden_dims[-1]),
                            nn.Tanh()
                            )            
                    
    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        embedding = self.bottleneck(result)

        return embedding

    def decode(self, z: Tensor):
        """
        Maps the given latent codes
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs):
        embedding = self.encode(input)
        output = self.decode(embedding)
        return  output
    

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
    

class VAEMY(nn.Module):
    def __init__(self,dim_au_in,dim_au_out=512,hidden_layers=[1024]):
        super(VAE, self).__init__()

        self.input_shape = dim_au_in
        self.bottleneck_shape = dim_au_out


        self.encode_h_layers = list()
        self.decode_h_layers = list()

        # The first hidden layer for encoder 
        self.encode_h_layers.append(
            nn.Linear(dim_au_in,hidden_layers[0])
        )

        # The last hidden layer for decoder
        self.decode_h_layers.append(
            nn.Linear(hidden_layers[0],dim_au_in)
        )

        # Intermediate hidden layers
        for i in range(0,len(hidden_layers)-1):
            self.encode_h_layers.append(
                nn.Linear(hidden_layers[i],hidden_layers[1+i])
            )
            
            self.decode_h_layers.insert(0,
                nn.Linear(hidden_layers[1+i],hidden_layers[i])
            )

        # The first hidden layer for decoder
        self.decode_h_layers.insert(0,
                nn.Linear(dim_au_out,hidden_layers[-1])
        )

        # bottle neck layers
        self.fc_mu = nn.Linear(hidden_layers[-1], dim_au_out)
        self.fc_logvar = nn.Linear(hidden_layers[-1], dim_au_out)



    def encode(self, x):
        l_input = x 
        for l in self.encode_h_layers:
            temp = l(l_input)
            l_input = F.relu(temp) 
        return self.fc_mu(l_input), self.fc_logvar(l_input)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        l_input = z 
        for l in self.decode_h_layers:
            temp = l(l_input)
            l_input = F.relu(temp) 

        #h3 = F.relu(self.fc3(z))
        return F.sigmoid(l_input)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE(nn.Module):
    def __init__(self,dim_au_in,dim_au_out=512):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(dim_au_in, 512)
        self.fc21 = nn.Linear(512, dim_au_out)
        self.fc22 = nn.Linear(512, dim_au_out)
        self.fc3 = nn.Linear(dim_au_out, 512)
        self.fc4 = nn.Linear(512, dim_au_in)

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

def vae_loss(recon_x, x, mu, logvar,reconstruction_function):
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

# class VAEBase(nn.Module):
#     def __init__(self,
#                  input_dim: int,
#                  latent_dim: int,
#                  hidden_dims: list = None,
#                  **kwargs):
                 
#         super(VAEBase, self).__init__()

#         self.latent_dim = latent_dim

#         modules = []
#         if hidden_dims is None:
#             hidden_dims = [512]
        
#         hidden_dims.insert(0,input_dim)

#         # Build Encoder
#         for i in range(1,len(hidden_dims)):
#             i_dim = hidden_dims[i-1]
#             o_dim = hidden_dims[i]

#             modules.append(
#                 nn.Sequential(
#                     nn.Linear(i_dim, o_dim),
#                     nn.BatchNorm2d(o_dim),
#                     nn.LeakyReLU())
#             )
#             #in_channels = h_dim

#         self.encoder = nn.Sequential(*modules)
#         self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
#         self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


#         # Build Decoder
#         modules = []

#         self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

#         hidden_dims.reverse()

#         for i in range(len(hidden_dims) - 2):
#             modules.append(
#                 nn.Sequential(
#                     nn.Linear(hidden_dims[i],
#                                        hidden_dims[i + 1],
#                     nn.BatchNorm2d(hidden_dims[i + 1]),
#                     nn.LeakyReLU())
#             ))


#         self.decoder = nn.Sequential(*modules)

#         self.final_layer = nn.Sequential(
#                             nn.Linear(hidden_dims[-2],
#                                        hidden_dims[-1],
#                             nn.Tanh())
#                             )  
    
#     def encode(self, input: Tensor):
#         """
#         Encodes the input by passing through the encoder network
#         and returns the latent codes.
#         :param input: (Tensor) Input tensor to encoder [N x C x H x W]
#         :return: (Tensor) List of latent codes
#         """
#         result = self.encoder(input)
#         #result = torch.flatten(result, start_dim=1)

#         # Split the result into mu and var components
#         # of the latent Gaussian distribution
#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)

#         return [mu, log_var]

#     def decode(self, z: Tensor):
#         """
#         Maps the given latent codes
#         onto the image space.
#         :param z: (Tensor) [B x D]
#         :return: (Tensor) [B x C x H x W]
#         """
#         result = self.decoder_input(z)
#         #result = result.view(-1, 512, 2, 2)
#         result = self.decoder(result)
#         result = self.final_layer(result)
#         return result

#     def reparameterize(self, mu: Tensor, logvar: Tensor):
#         """
#         Reparameterization trick to sample from N(mu, var) from
#         N(0,1).
#         :param mu: (Tensor) Mean of the latent Gaussian [B x D]
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
#         :return: (Tensor) [B x D]
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu

#     def forward(self, input: Tensor, **kwargs):
#         mu, log_var = self.encode(input)
#         z = self.reparameterize(mu, log_var)
#         return  [self.decode(z), input, mu, log_var]

#     def loss_function(self,
#                       *args,
#                       **kwargs) -> dict:
#         """
#         Computes the VAE loss function.
#         KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
#         :param args:
#         :param kwargs:
#         :return:
#          M_N = self.params['batch_size']/ self.num_train_imgs,
#         """
#         recons = args[0]
#         input = args[1]
#         mu = args[2]
#         log_var = args[3]

#         kld_weight = kwargs['M_N'] 
#         # Account for the minibatch samples from the dataset
#         # M_N = self.params['batch_size']/ self.num_train_imgs,
#         recons_loss =F.mse_loss(recons, input)


#         kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

#         loss = recons_loss + kld_weight * kld_loss
#         return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

#     def sample(self,
#                num_samples:int,
#                current_device: int, **kwargs):
#         """
#         Samples from the latent space and return the corresponding
#         image space map.
#         :param num_samples: (Int) Number of samples
#         :param current_device: (Int) Device to run the model
#         :return: (Tensor)
#         """
#         z = torch.randn(num_samples,
#                         self.latent_dim)

#         z = z.to(current_device)

#         samples = self.decode(z)
#         return samples

#     def generate(self, x: Tensor, **kwargs):
#         """
#         Given an input image x, returns the reconstructed image
#         :param x: (Tensor) [B x C x H x W]
#         :return: (Tensor) [B x C x H x W]
#         """

#         return self.forward(x)[0]