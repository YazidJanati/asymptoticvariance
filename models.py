import numpy as np 
import torch
from torch import nn
from torch.distributions import Normal, Categorical, Gumbel
import time

pi = torch.tensor(np.pi)

class LinGauss:
    
    def __init__(self, a, b, sigma_y):
        self.a = torch.tensor(a)
        self.b = torch.tensor(b)
        sigma_y = torch.tensor(sigma_y)
    
    def log_prob(self, xt_1, xt):
        return - ((xt -  self.a*xt_1)**2 + (2*pi).log())/2
    
    def log_likelihood(self, yt, xt):
        return Normal(0,1).log_prob(yt - self.b * xt)
    
    def kalman_filter(self, obs):
        Qt = 1 / 2
        mt = Qt * obs[0]
        m_preds = [0]
        m_filters = [mt]
        for t in range(1,len(obs)):
            m_preds.append(self.a * m_filters[-1])
            Et = self.a ** 2 * Qt + 1
            Qt = Et * (1 - Et / (1 + Et))
            m_filters.append((1 - Et/(1 + Et)) * self.a * m_filters[-1] + Et * obs[t] / (1 + Et))
        return m_filters, m_preds

    
    def sample_transition(self, xt_1):
        return torch.randn_like(xt_1) + self.a * xt_1
    
    def sample_prior(self, N_samples):
        return torch.randn(N_samples, 1)
    
    def sample(self, time_steps):
        states = torch.randn(1)
        obs = torch.randn(1) + states[-1]
        for t in range(time_steps):
            states = torch.cat([states, self.sample_transition(torch.tensor([states[-1]]))])
            obs = torch.cat([obs, torch.randn(1) + self.b * states[-1]])

        return states, obs
    
class StochasticVol:
    
    def __init__(self, phi, beta, sigma):
        self.phi   = torch.tensor(phi)
        self.beta  = torch.tensor(beta)
        self.sigma = torch.tensor(sigma)

    def log_prob(self, xt_1, xt):
        return - ( ((xt -  self.phi*xt_1) / self.sigma)**2 + (2*pi).log() + 2*self.sigma.log() ) / 2
    
    def log_likelihood(self, yt, xt):
        std = self.beta * (xt/2).exp()
        return - ((yt / std)**2 + (2*pi).log() + 2*std.log() ) / 2
    
    def sample_transition(self, xt_1):
        return self.sigma * torch.randn_like(xt_1) + self.phi * xt_1
    
    def sample_prior(self, N_samples):
        return torch.randn(N_samples, 1)
    
    def sample_obs(self, xt):
        return self.beta * (xt / 2).exp() * torch.randn(1)

    def sample_xtyt(self,xt_1):
        xt = self.sample_transition(xt_1)
        return xt, self.sample_obs(xt)

    def sample(self, time_steps):
        states = torch.randn(1)
        obs    = self.beta * (states / 2).exp() * torch.randn(1)

        for t in range(time_steps):
            states = torch.cat([states, self.sample_transition(torch.tensor([states[-1]]))])
            obs    = torch.cat([obs, self.beta * (states[-1] / 2).exp() * torch.randn(1)])
        
        return states, obs