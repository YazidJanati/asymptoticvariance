import numpy as np 
import torch
from torch import nn
from torch.distributions import Normal, Categorical, Gumbel
import pickle as pkl 

pi = torch.tensor(np.pi)

class SMC(nn.Module):
    
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        
    def initialize(self, y0, N):
        xi0 = self.model.sample_prior(N).to(self.device)
        log_wt = self.model.log_likelihood(xi0, y0).to(self.device)
        return xi0, log_wt

    def step(self, xit_1, wt_1, yt):
        N = len(xit_1)
        At_1 = Categorical(wt_1.flatten()).sample((N,))
        xt = self.model.sample_transition(xit_1[At_1]).to(self.device)
        log_wt = self.model.log_likelihood(yt, xt).to(self.device)
        return xt, log_wt, At_1

    def _backward_weights(self, xit_1, xit, log_wt_1):
        bw = log_wt_1.reshape(-1,1) + self.model.log_prob(xit_1.reshape(-1,1), xit.reshape(1,-1))
        bw = bw.softmax(0)
        return bw

    def marginalSM_step(self, bw, log_wt, TN):
        TN = (bw.T @ TN).reshape(-1,1)
        marginal_smooth = (log_wt.softmax(0).flatten() * TN.flatten())
        return marginal_smooth.sum().item(), TN
    