- add critic model, to be able to judge quality of state without completing full pass
- add safe policy, and then during rollouts use it every $N$ steps for $M$ steps (e.g. every 100 for 10). safe policy will just return the state to something reasonable, e.g. uniform distribution between all stocks and cash. also cut gradients after safe policy is used, then one sub-episode of target + safe policy can be used for updates, and there will be no (well, less) gradient explosion and also no problems with getting stuck at boundaries, etc.
- correlation between wealth and alpha
- other features to use by agents
- mech interp:
    - (anti-dropout & (L_1 or L_log penalty)) for sparsification
    - replacing layers, or their parts, or even specific paths, with simpler functions
- G.pt for ESN
- to trade-off between training on all stocks vs. one stock, make a strategy with a variable number of stocks, and train on (small) random subsets of stocks
- find small basis for EMA, TEMA, SMA, SMMA, Moving Linear Regression, TRIX, etc.
- plot cumulative volumes per price level
- one indicator to rule them all. MLP-like штука, на входе свёртки с затухающими экспонентами и синусоидами (тоже экспоненты по сути, лол), дальше слои вида (a_0 x_0 + ... a_n x_n) / (1 + abs(b_0 x_0 + ... b_n x_n + c_0 abs(x_0) + ... + c_n abs(x_n) )).
- optimal trading via dynamic programming, then learn it in supervised fashion
- many elementary RL environments to benchmark various aspects (relevant to trading). e.g. there is basically no exploration problem in trading, it's almost supervised learning, so some problems that occur e.g. in Atari might be irrelevant.
    - environment where agent just needs to always keep certain cash ratio
    - environment where next price is given as one of the signals (and others are noise)
    - environment where all signals are noise, but the price process is mean-reverting (e.g. MA(q))
    - environment where all signals are noise, but the price process is trend-following (e.g. prefix sums of random walk, in log-space)
    - environment where optimal action is encoded as XOR of k signals
    - envrironment with noise signals where at t * i period i-th stock price grows by a lot (so that the optimal strategy, without knowing exactly which stock will grow, is to diversify,
    and diversify in terms of money, not shares)
- select the best practices for optimization
    - Lookahead 
    - Cautious optimizers 
    - Weight averaging 
    - Sphere GPT
    - x-LSTM
    - other stuff from the saved ones