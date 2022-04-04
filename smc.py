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
        """
        Initialization of the bootstrap algorithm by drawing N i.i.d. samples from the
        initial distribution.

        Parameters
        ----------
        y0: 1x1 torch.tensor
            initial observation.
        
        N: int
           number of samples.

        Returns
        -------
        xi0: Nx1 torch.tensor
             initial particles.
    
        log_wt: Nx1 torch.tensor
                log weights associated to the particles.
        """
        xi0 = self.model.sample_prior(N).to(self.device)
        log_wt = self.model.log_likelihood(xi0, y0).to(self.device)
        return xi0, log_wt

    def step(self, xit_1, wt_1, yt):
        """
        Performs one step of the bootstrap algorithm: selection -> mutation -> correction

        Parameters
        ----------
        xit_1 : Nx1 torch.tensor
                particles at step t-1
        
        wt_1  : Nx1 torch.tensor
                weights of said particles at step t-1
        
        yt : 1x1 torch.tensor
             current observation

        Returns
        -------
        xt: Nx1 torch.tensor
            particles at step t
        
        log_wt: Nx1 torch.tensor
                log weights of said particles at step t
        
        At_1: Nx1 torch.tensor
              ancestor at step t-1 of current particles
        """
        N = len(xit_1)
        At_1 = Categorical(wt_1.flatten()).sample((N,))
        xt = self.model.sample_transition(xit_1[At_1]).to(self.device)
        log_wt = self.model.log_likelihood(yt, xt).to(self.device)
        return xt, log_wt, At_1

    def _backward_weights(self, xit_1, xit, log_wt_1):
        """
        Computes the backward weights.

        Returns
        -------
        bw: NxN torch.tensor
            backward weights. bw[i,j] = \beta^\BS _t(j,i) (see main paper)
        """
        bw = log_wt_1.reshape(-1,1) + self.model.log_prob(xit_1.reshape(-1,1), xit.reshape(1,-1))
        bw = bw.softmax(0)
        return bw

    def marginalSM_step(self, bw, log_wt, TN):
        """
        Performs one step of FFBS marginal smoothing.

        Parameters
        ----------
        TN: NxN torch.tensor
            backward statistic (see section 5.1 of the main paper)

        Returns
        -------
        marginal smooth: float
                         marginal smoothing at time t
        TN: Nx1 torch.tensor
            backward statistic at time t
        """
        TN = (bw.T @ TN).reshape(-1,1)
        marginal_smooth = (log_wt.softmax(0).flatten() * TN.flatten())
        marginal_smooth = marginal_smooth.sum().item()
        return marginal_smooth, TN
    