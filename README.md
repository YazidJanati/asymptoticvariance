Companion code for the paper "Variance estimation for Sequential Monte Carlo Algorithms: a backward sampling approach" by Yazid Janati El Idrissi, Sylvain Le Corff and Yohan Petetin. 


**Example**

First import the relevant functions.

```python
import numpy as np 
import torch
from smc import *
from variance_estimators import *
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Generate states and observations from a state space model (Stochastic volatility model here).

```python
stovol = StochasticVol(phi = .975, beta = .641, sigma = .165)
smc  = SMC(stovol, device)
time_steps = 100

state, obs = smc.model.sample(time_steps)
```
Then run an SMC and compute the variances online. 

```python
N = 500
lag = 20 
timesteps = len(obs)
yt = obs[0]

eves = torch.tensor(range(N)).reshape(-1,1)
enochs = eves.clone()

xit, log_wt = smc.initialize(yt, N)
tau0 = (torch.ones(N,N) - torch.eye(N)).to(device)
ptau0  = tau0.clone()
tau0_, sum_es = initVBS(N, device)

for t in range(1, 4):
    yt = obs[t]
    xit_1, log_wt_1 = xit, log_wt
    xit, log_wt, At_1 = smc.step(xit, log_wt.softmax(0), yt)
    bw = smc._backward_weights(xit_1, xit, log_wt_1)
    eves   = eves[At_1]

    if t == 1: 
        enochs = torch.cat([enochs[At_1,1:], At_1.reshape(-1,1)], 1)
    elif 1 < t <= lag: 
        enochs = torch.cat([enochs[At_1,:], At_1.reshape(-1,1)], 1)
    elif t > lag: 
        enochs = torch.cat([enochs[At_1, 1:], At_1.reshape(-1,1)], 1)
        
    asymptvarBS, tau0 = VBS_filter(xit, log_wt.softmax(0), bw, tau0, M = 3, timestep = t, paris = False)
    asymptvarBSp, ptau0 = VBS_filter(xit, log_wt.softmax(0), bw, ptau0, M = 3, timestep = t, paris = True)
    tbt, sum_es, tau0_ = tbtVBS(xit, bw, log_wt_1.softmax(0), sum_es, tau0_, t)
    LCLE = lagged_CLE(xit, enochs)
```

