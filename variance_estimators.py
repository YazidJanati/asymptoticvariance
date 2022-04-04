import numpy as np 
import torch
from torch import nn
from torch.distributions import Normal, Categorical

"""
[1] Olsson, Jimmy, and Randal Douc. "Numerically stable online estimation of variance in particle filters." Bernoulli 25.2 (2019): 1504-1535.
[2] Chan, Hock Peng, and Tze Leung Lai. "A general theory of particle filters in hidden Markov models and some applications." The Annals of Statistics 41.6 (2013): 2877-2904.

"""

def tau0_(bw, tau0):
    """
    Computes the statistic Tau^0 _t (see 4.2 in 
    the main paper).

    Parameters
    ----------
    bw: NxN torch.tensor
        backward weights. bw[i,j] = \beta^\BS _t(j,i) (see section 4.1)

    tau0: NxN torch.tensor
          Tau^0 _{t-1}. tau0_t[k,l] = Tau^0 _t(k,l).
    """
    N = bw.shape[0]
    tau0_t = bw.T @ tau0 @ bw
    tau0_t[range(N), range(N)] = 0.
    return tau0_t

def ptau0_(bw, ptau0, M):
    """
    Computes the parisian version of Tau^0 _t (see 4.4
    in the main paper).

    Parameters
    ----------
    M: int
       number of sampled indices.
    """
    N = bw.shape[0]
    idxs = Categorical(bw.T).sample((M,)) 
    ptau0_t = ptau0[idxs.T.unsqueeze(-1), idxs]
    ptau0_t = ptau0_t.mean(-2)
    ptau0_t[range(N), range(N)] = 0.
    return ptau0_t

def initVBS(N, device):
    """
    Computes Tau^0 _0 and Tau^1 _0.
    """
    tau0 = torch.ones(N,N) - torch.eye(N)
    taue0 = torch.eye(N)
    return tau0.to(device), taue0.to(device)

def VBS(xit, bw, tau0, M, timestep, paris = False):
    """
    Computes the simpler estimator defined in 
    section 4.3 of the paper, for the predictive mean.

    Parameters
    ----------
    xit: Nx1 torch.tensor
         particles at time t

    bw: NxN torch.tensor
        backward weights. 

    tau0: NxN torch.tensor
           tau^0 _{t-1}
    
    M: int
       number of sampled indices if paris is used.

    timestep: int
              current timestep.

    paris: bool

    Returns
    -------
    vbs: 1x1 torch.tensor
         variance estimate at time t
    
    tau0_t: NxN torch.tensor
            updated statistic, tau^0 _t
    """
    N = bw.shape[0]
    ctrd_h = xit.reshape(-1,1) - xit.mean(0)
    if paris == True:
        tau0_t  = ptau0_(bw, tau0, M)
    else:
        tau0_t  = tau0_(bw, tau0)
    Q = tau0_t * (ctrd_h * ctrd_h.T)
    cteN = N / (N - 1)
    vbs  = - cteN ** (timestep +1) * Q.sum() / N
    return vbs, tau0_t

def VBS_filter(xit, wt, bw, tau0, M, timestep, paris = False):
    """
    Computes the simpler estimator for the filtering mean. 
    """
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
    """
    Computes the statistic S_t defined in section 4.2 
    of the main paper.

    Parameters
    ----------
    sumtaues: NxN torch.tensor
              S_{t-1}. sumtaues[i,j] = S_{t-1}(j,i).   

    Returns
    -------
    sumtaues_t: NxN torch.tensor
                S_t

    tau0_t : NxN torch.tensor
    tau_et : NxN torch.tensor
             Tau^{e_t} _t.     
    """
    N = bw.shape[0]
    N_ = range(N)
    tau_et  = bw.T @ tau0 @ wt_1.reshape(-1,1)
    tau_et  = torch.diag(tau_et.flatten())
    sumtaues_t = bw.T @ sumtaues @ bw 
    sumtaues_t[N_, N_] = 0.
    sumtaues_t = sumtaues_t + tau_et
    tau0_t  = bw.T @ tau0 @ bw 
    tau0_t[N_, N_] = 0.
    return sumtaues_t, tau0_t, tau_et

def tbtVBS(xit, bw, wt_1, sumtaues, tau0, timestep):
    """
    Computes the term by term estimator.
    """
    N = bw.shape[0]
    ctrd_h = xit.reshape(-1,1) - xit.mean(0)
    cteN   = N / (N-1)
    sumtaues_t, tau0_t, _ = sumtaues_(bw, wt_1, sumtaues, tau0)
    Qes_Q0 = N * cteN**(timestep) * sumtaues_t - cteN**(timestep + 1) * tau0_t * (timestep + 1)
    tbtest = (Qes_Q0 * (ctrd_h * ctrd_h.T)).mean()
    return tbtest, sumtaues_t, tau0_t 

def lagged_CLE(xit, enochs):
    """
    Computes the lagged CLE of Olsson & Douc [1].
    """
    N = xit.shape[0]
    var, predictive = 0, xit.mean(0)
    for enoch in torch.unique(enochs[:,0]):
        idxs = (enochs[:,0] == enoch).nonzero()[:,0]
        var += (xit[idxs] - predictive).sum() ** 2
    return var / N

def CLE(xit, eves):
    """
    Computes the CLE of Chan & Lai [2].
    """
    N = xit.shape[0]
    var, predictive = 0, xit.mean(0)
    for eve in torch.unique(eves):
        idxs = (eves == eve).nonzero()[:,0]
        var += (xit[idxs] - predictive).sum() ** 2
    return var / N

def margsmoothVBS(timestep, wt, bw, TN, sumtaues, tau_et, S1, S2, margsmooth):
    """
    Computes the variance estimator of the marginal smoothing estimator.
    """
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

