import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPS = 1e-5

def _two_pi_like(x):  # scalar on same device/dtype
    return x.new_tensor(2.0 * math.pi)
# PI = torch.from_numpy(np.asarray(np.pi))

def log_categorical(x, p, num_classes=256, reduction=None):
    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))
    if reduction == 'mean':
        return torch.mean(log_p, list(range(1, len(x.shape))))
    elif reduction == 'sum':
        return torch.sum(log_p, list(range(1, len(x.shape))))
    else:
        return log_p
    
def log_bernoulli(x, p, reduction=None):
    pp = torch.clamp(p, EPS, 1. - EPS)
    log_p = x * torch.log(pp) + (1. - x) * torch.log(1. - pp)
    if reduction == 'mean':
        return torch.mean(log_p, list(range(1, len(x.shape))))
    elif reduction == 'sum':
        return torch.sum(log_p, list(range(1, len(x.shape))))
    else:
        return log_p
    
def log_normal_diag(x, mu, log_var, reduction=None):
    # D = x.shape[1]
    # log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    # if reduction == 'mean':
    #     return torch.mean(torch.sum(log_p, list(range(1, len(x.shape)))))
    # elif reduction == 'sum':
    #     return torch.sum(torch.sum(log_p, list(range(1, len(x.shape)))))
    # else:
    #     return log_p
    log_p = -0.5 * (torch.log(_two_pi_like(x)) + log_var + torch.exp(-log_var) * (x - mu) ** 2)
    if reduction == 'mean':
        return torch.sum(log_p, dim=tuple(range(1, log_p.ndim))).mean(0)
    elif reduction == 'sum':
        return torch.sum(log_p, dim=tuple(range(1, log_p.ndim))).sum(0)
    else:
        return log_p    

def log_standard_normal(x, reduction=None):
    # D = x.shape[1]
    # log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
    # if reduction == 'mean':
    #     return torch.mean(log_p, list(range(1, len(x.shape))))
    # elif reduction == 'sum':
    #     return torch.sum(log_p, list(range(1, len(x.shape))))
    # else:
    #     return log_p
    log_p = -0.5 * (torch.log(_two_pi_like(x)) + x ** 2)
    if reduction == 'mean':
        return torch.sum(log_p, dim=tuple(range(1, log_p.ndim))).mean(0)
    elif reduction == 'sum':
        return torch.sum(log_p, dim=tuple(range(1, log_p.ndim))).sum(0)
    else:
        return log_p        

class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super(Encoder, self).__init__()
        self.encoder = encoder_net

    @staticmethod
    def reparameterization(mu, log_var):
        # return is the result of the reparameterization trick
        # The formula is the following:
        # z = mu + std * epsilon
        # epsilon ~ Normal(0,1)
        # First, we need to get std from log-variance.
        std = torch.exp(0.5 * log_var)
        # Second, we sample epsilon from Normal(0,1).
        eps = torch.randn_like(std)
        # The final output
        return mu + std * eps
    
    def encode(self, x):
        # return is means and log-variances (or variances)
        # First, we calculate the output of the encoder netowork of size 2M.
        h_e = self.encoder(x)
        # Second, we must divide the output to the mean and the log-variance.
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        return mu_e, log_var_e

    def sample(self, x=None, mu_e=None, log_var_e=None):
        # return is a sample from a Gaussian variational posterior
        # If we don't provide a mean and a log-variance, we must first calcuate it:
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.encode(x)
        # Or the final sample
        else:
        # Otherwise, we can simply apply the reparameterization trick!
            if (mu_e is None) or (log_var_e is None):
                raise ValueError('mu and log-var can`t be None!')
        z = self.reparameterization(mu_e, log_var_e)
        return z

    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        # return is the log-probability of a Gaussian with diagonal covariance matrix
        # If we provide x alone, then we can calculate a corresponsing sample:
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        else:
        # Otherwise, we should provide mu, log-var and z!
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError('mu, log-var and z can`t be None!')
        return log_normal_diag(z, mu_e, log_var_e)
    
    def forward(self, x, type='log_prob'):
        # return is the log-probability
        assert type in ['encode', 'log_prob', 'sample'], 'Type could be either encode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x)
        elif type == 'encode':
            return self.encode(x)
        else:
            return self.sample(x)

class Decoder(nn.Module):
    def __init__(self, decoder_net):
        super(Decoder, self).__init__()
        # The decoder network.
        self.decoder = decoder_net
    
    def decode(self, z):
        # return is parameters of the conditional distribution (decoder)
        # First, we apply the decoder network.
        h_d = self.decoder(z)
        mu, log_var = torch.chunk(h_d, 2, dim=1)  # split into mean and logvar
        log_var = torch.clamp(log_var, min=-4, max=4)
        return [mu, log_var]        
    
    def sample(self, z):
        # return is a sample
        outs = self.decode(z)
        mu, log_var = outs
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        x_new = mu + eps * std
        return x_new

    def log_prob(self, x, z):
        # return is the log-probability
        outs = self.decode(z)
        mu, log_var = outs
        log_p = log_normal_diag(x, mu, log_var)
        log_p = torch.sum(log_p, dim=1)
        return log_p
    
    def forward(self, z, x=None, type='log_prob'):
        # return is the log-probability
        assert type in ['decode', 'log_prob'], 'Type could be either decode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x, z)
        else:
            return self.sample(z)

class Prior(nn.Module):
    def __init__(self, L, num_components, multiplier=1):
        super(Prior, self).__init__()
        self.L = L
        self.num_components = num_components
        # params
        self.means = nn.Parameter(torch.randn(num_components, self.L) * multiplier)
        self.logvars = nn.Parameter(torch.randn(num_components, self.L))
        # mixing weights
        self.w = nn.Parameter(torch.zeros(num_components, 1, 1))

    # def sample(self, batch_size):
    #     # return is a sample
    #     # mu, log_var
    #     means, logvars = self.means, self.logvars
    #     means = means.to(device)
    #     logvars = logvars.to(device)
    #     # mixing probabilities
    #     w = F.softmax(self.w, dim=0)
    #     w = w.squeeze().to(device)
    #     # pick components
    #     indexes = torch.multinomial(w, batch_size, replacement=True).to(device)
    #     # means and logvars
    #     eps = torch.randn(batch_size, self.L).to(device)
    #     for i in range(batch_size):
    #         indx = indexes[i]
    #         if i == 0:
    #             z = means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])
    #         else:
    #             z = torch.cat((z, means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])), 0)
    #     return z

    def sample(self, batch_size):
        means, logvars = self.means.to(device), self.logvars.to(device)
        stds = torch.exp(0.5 * logvars)
        w = F.softmax(self.w, dim=0).squeeze().to(device)
        idx = torch.multinomial(w, batch_size, replacement=True)
        eps = torch.randn(batch_size, self.L, device=device)
        z = means[idx] + eps * stds[idx]
        return z    

    def log_prob(self, z):
        # return is the log-probability
        # mu, lof_var
        means, logvars = self.means, self.logvars
        # mixing probabilities
        w = F.softmax(self.w, dim=0)
        # log-mixture-of-Gaussians
        z = z.unsqueeze(0) # 1 x B x L
        means = means.unsqueeze(1) # K x 1 x L
        logvars = logvars.unsqueeze(1) # K x 1 x L
        log_p = log_normal_diag(z, means, logvars) + torch.log(w) # K x B x L
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False) # B x L
        return log_prob      
    
# This class combines Encoder, Decoder and Prior.
class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior, L=16):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        # self.beta = 0.1

    def sample(self, batch_size=64):
        # return is a sample
        # return is the Negative ELBO
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)

    def forward(self, x, reduction='mean'):
        # self.beta = min(10, self.beta + 0.2)
        # encoder
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(x=x, mu_e=mu_e, log_var_e=log_var_e)

        # ELBO
        RE = self.decoder.log_prob(x, z)
        KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)

        error = 0
        if np.isnan(RE.detach().cpu().numpy()).any():
            print("\nRE is nan")
            error = 1
        if np.isnan(KL.detach().cpu().numpy()).any():
            print("\nKL is nan")
            error = 1

        if error == 1:
            raise ValueError()

        if reduction == 'sum':
            return -(RE + KL).sum(), RE.mean(), KL.mean()
        else:
            return -(RE + KL).mean(), RE.mean(), KL.mean()    