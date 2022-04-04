import numpy as np 
import torch
from torch import nn
from torch.distributions import Normal, Categorical, Gumbel

def tau0_(bw, tau0):
    N = bw.shape[0]
    tau0_t = bw.T @ tau0 @ bw
    #tau0_t = tau0_t - torch.diag(tau0_t) * torch.eye(tau0_t.shape[0]).to(bw.device)
    tau0_t[range(N), range(N)] = 0.
    return tau0_t

def ptau0_(bw, ptau0, M):
    N = bw.shape[0]
    idxs = Categorical(bw.T).sample((M,)) 
    ptau0_t = ptau0[idxs.T.unsqueeze(-1), idxs]
    ptau0_t = ptau0_t.mean(-2)
    #ptau0_t = ptau0_t - torch.diag(ptau0_t) * torch.eye(ptau0_t.shape[0]).to(bw.device)
    ptau0_t[range(N), range(N)] = 0.
    return ptau0_t

def initVBS(N, device):
    tau0 = torch.ones(N,N) - torch.eye(N)
    taue0 = torch.eye(N)
    return tau0.to(device), taue0.to(device)

def VBS(xit, bw, tau0, M, timestep, paris = False):
    N = bw.shape[0]
    ctrd_h = xit.reshape(-1,1) - xit.mean(0)
    if paris == True:
        tau0_t  = ptau0_(bw, tau0, M)
    else:
        tau0_t  = tau0_(bw, tau0)
    Q = tau0_t * (ctrd_h * ctrd_h.T)
    cteN = N / (N - 1)
    return - cteN ** (timestep +1) * Q.sum() / N, tau0_t

def VBS_filter(xit, wt, bw, tau0, M, timestep, paris = False):
    N = bw.shape[0]
    filter_ = (wt * xit).sum()
    ctrd_h = xit.reshape(-1,1) - filter_
    if paris == True:
        tau0_t  = ptau0_(bw, tau0, M)
    else:
        tau0_t  = tau0_(bw, tau0)
    Q = tau0_t * ((wt * ctrd_h) * (wt * ctrd_h).T)
    cteN = N / (N - 1)
    return - N * cteN ** (timestep +1) * Q.sum() , tau0_t

def sumtaues_(bw, wt_1, sumtaues, tau0):
    N = bw.shape[0]
    N_ = range(N)
    eye = torch.eye(N).to(bw.device)
    tau_et  = bw.T @ tau0 @ wt_1.reshape(-1,1)
    tau_et  = torch.diag(tau_et.flatten())
    sumtaues_t = bw.T @ sumtaues @ bw 
    sumtaues_t[N_, N_] = 0.
    sumtaues_t = sumtaues_t + tau_et
    tau0_t  = bw.T @ tau0 @ bw 
    #tau0_t  = tau0_t - torch.diag(tau0_t) * eye
    tau0_t[N_, N_] = 0.
    return sumtaues_t, tau0_t, tau_et

def tbtVBS(xit, bw, wt_1, sumtaues, tau0, timestep):
    #term by term estimator
    N = bw.shape[0]
    ctrd_h = xit.reshape(-1,1) - xit.mean(0)
    cteN   = N / (N-1)
    sumtaues_t, tau0_t, _ = sumtaues_(bw, wt_1, sumtaues, tau0)
    Qes_Q0 = N * cteN**(timestep) * sumtaues_t - cteN**(timestep + 1) * tau0_t * (timestep + 1)
    tbtest = (Qes_Q0 * (ctrd_h * ctrd_h.T)).mean()
    return tbtest, sumtaues_t, tau0_t 

def lagged_CLE(xit, enochs):
    N = xit.shape[0]
    var, predictive = 0, xit.mean(0)
    for enoch in torch.unique(enochs[:,0]):
        idxs = (enochs[:,0] == enoch).nonzero()[:,0]
        var += (xit[idxs] - predictive).sum() ** 2
    return var / N

def CLE(xit, eves):
    N = xit.shape[0]
    var, predictive = 0, xit.mean(0)
    for eve in torch.unique(eves):
        idxs = (eves == eve).nonzero()[:,0]
        var += (xit[idxs] - predictive).sum() ** 2
    return var / N

def margsmoothVBS(timestep, wt, bw, TN, sumtaues, tau_et, S1, S2, margsmooth):
    N = bw.shape[0]
    TN, wt = TN.reshape(-1,1), wt.reshape(-1,1)
    #sumtaues, tau0, tau_et = sumtaues_(bw, wt_1, sumtaues, tau0)
    eye = torch.eye(N).to(bw.device)
    S1_t = bw.T @ S1 @ bw
    S2_t = bw.T @ S2 @ bw
    S1_t = S1_t - torch.diag(S1_t) * eye
    S2_t = S2_t - torch.diag(S2_t) * eye
    S1_t = S1_t + tau_et * (TN * TN.T)
    S2_t = S2_t + tau_et * (TN + TN.T)
    Sbar = (wt * wt.T) * (S1_t - margsmooth * S2_t + margsmooth**2 * sumtaues)
    cteN = N / (N-1)
    return N**3 * cteN**(timestep) * Sbar.mean(), S1_t, S2_t

