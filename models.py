import torch
from torch import nn, optim, zeros_like
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch import Tensor
from torch.nn import Dropout

from gae.layers import GraphConvolution,InnerProductDecoder

#import scipy.io as sio
import numpy as np
import pandas as pd
from copy import deepcopy

# Model of AE
class AEBase(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(AEBase, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)

        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    #nn.ReLU(),
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
                    #nn.ReLU(),
                    nn.Dropout(drop_out))
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-2],
                                       hidden_dims[-1])
                                       ,nn.Sigmoid()
                            )
        # self.feature_extractor =nn.Sequential(
        #     self.encoder,
        #     self.bottleneck
        # )            
                    
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

# Model of Predictor
class Predictor(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=1,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(Predictor, self).__init__()

        modules = []

        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)

        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_out))
            )
            #in_channels = h_dim

        self.predictor = nn.Sequential(*modules)
        #self.output = nn.Linear(hidden_dims[-1], output_dim)

        self.output = nn.Sequential(
                            nn.Linear(hidden_dims[-1],
                                       output_dim),
                                       nn.Sigmoid()
                            )            

    def forward(self, input: Tensor, **kwargs):
        embedding = self.predictor(input)
        output = self.output(embedding)
        return  output
    
# Model of Pretrained P
class PretrainedPredictor(AEBase):
    def __init__(self,
                 # Params from AE model
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3,
                 ### Parameters from predictor models
                 pretrained_weights=None,                 
                 hidden_dims_predictor=[256],
                 drop_out_predictor=0.3,
                 output_dim = 1,
                 freezed = False):
        
        # Construct an autoencoder model
        AEBase.__init__(self,input_dim,latent_dim,h_dims,drop_out)
        
        # Load pretrained weights
        if pretrained_weights !=None:
            self.load_state_dict((torch.load(pretrained_weights)))
        
        ## Free parameters until the bottleneck layer
        if freezed == True:
            for p in self.parameters():
                print("Layer weight is freezed:",format(p.shape))
                p.requires_grad = False
                # Stop until the bottleneck layer
                if p.shape.numel() == self.latent_dim:
                    break
        # Only extract encoder
        del self.decoder
        del self.decoder_input
        del self.final_layer

        self.predictor = Predictor(input_dim=self.latent_dim,
                 output_dim=output_dim,
                 h_dims=hidden_dims_predictor,
                 drop_out=drop_out_predictor)
        
        # self.feature_extractor = nn.Sequential(
        #     self.encoder,
        #     self.bottleneck
        # )
        

    def forward(self, input, **kwargs):
        embedding = self.encode(input)
        output = self.predictor(embedding)
        return  output
    

def vae_loss(recon_x, x, mu, logvar,reconstruction_function,weight=1):
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
    return BCE + KLD * weight

class VAEBase(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(VAEBase, self).__init__()

        self.latent_dim = latent_dim

        modules = []
    
        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)
        
        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
            )
            #in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


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
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-2],
                                       hidden_dims[-1],
                            nn.Sigmoid())
                            ) 
        # self.feature_extractor = nn.Sequential(
        #     self.encoder,
        #     self.fc_mu
        # )
    
    def encode_(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        #result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        mu, log_var = self.encode_(input)
        z = self.reparameterize(mu, log_var)

        return z

    def decode(self, z: Tensor):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):
        mu, log_var = self.encode_(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
         M_N = self.params['batch_size']/ self.num_train_imgs,
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] 
        # Account for the minibatch samples from the dataset
        # M_N = self.params['batch_size']/ self.num_train_imgs,
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class PretrainedVAEPredictor(VAEBase):
    def __init__(self,
                 # Params from AE model
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3,
                 ### Parameters from predictor models
                 pretrained_weights=None,                 
                 hidden_dims_predictor=[256],
                 drop_out_predictor=0.3,
                 output_dim = 1,
                 freezed = False):
        
        # Construct an autoencoder model
        VAEBase.__init__(self,input_dim,latent_dim,h_dims,drop_out)
        
        # Load pretrained weights
        if pretrained_weights !=None:
            self.load_state_dict((torch.load(pretrained_weights)))
        
        ## Free parameters until the bottleneck layer
        if freezed == True:
            for p in self.parameters():
                print("Layer weight is freezed:",format(p.shape))
                p.requires_grad = False
                # Stop until the bottleneck layer
                if p.shape.numel() == self.latent_dim:
                    break
        # Only extract encoder
        del self.decoder
        del self.decoder_input
        del self.final_layer

        self.predictor = Predictor(input_dim=self.latent_dim,
                 output_dim=output_dim,
                 h_dims=hidden_dims_predictor,
                 drop_out=drop_out_predictor)

        # self.feature_extractor = nn.Sequential(
        #     self.encoder,
        #     self.fc_mu
        # )

    def forward(self, input, **kwargs):
        embedding = self.encode(input)
        output = self.predictor(embedding)
        return  output

class GAEBase(nn.Module):
    
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):
                 
        super(GAEBase, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        hidden_dims = deepcopy(h_dims)
        
        hidden_dims.insert(0,input_dim)

        # Build Encoder
        for i in range(1,len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    GraphConvolution(i_dim, o_dim,drop_out, act=lambda x: x),
                    #nn.BatchNorm1d(o_dim),
                    #nn.ReLU()
                    #nn.Dropout(drop_out)
                    )
            )
            #in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.bottleneck = GraphConvolution(hidden_dims[-1], latent_dim,drop_out, act=lambda x: x)

        # Build Decoder
        self.decoder = InnerProductDecoder(drop_out, act=lambda x: x)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        embedding,adj = self.bottleneck(result)

        return embedding

    def decode(self, z):
        """
        Maps the given latent codes
        """
        result = self.decoder(z)
        return result

    def forward(self, input):
        embedding = self.encode(input)
        output = self.decode(embedding)
        return  output


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GCNPredictor(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout,
                pretrained_weights=None,                 
                hidden_dims_predictor=[256],
                drop_out_predictor=0.3,
                output_dim = 1,
                freezed = False):
        super(GCNPredictor, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

        self.predictor = Predictor(input_dim=hidden_dim2,
            output_dim=output_dim,
            h_dims=hidden_dims_predictor,
            drop_out=drop_out_predictor)


    def encode(self, x, adj):
        hidden1,_adj = self.gc1((x, adj))
        result,_ = self.gc2((hidden1, adj))
        return result

    def forward(self, x, adj, encode=False):
        z = self.encode(x, adj)
        result = self.predictor(z)
        return result


def g_loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD