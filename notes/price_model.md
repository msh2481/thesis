# Market Model Exploration

I want to explore ideas about markets and analyze market data from different angles to discover market inefficiencies that could lead to profitable trading strategies.

## Synthetic Data Generation Model

The goal is to generate synthetic data that mimics real market behavior to understand key market forces and their interactions. Here's the proposed model:

### 1. Fair Price Process
Let $F(t)$ be the fair price, which in log-space follows either:
- A Gaussian random walk with scale $s$, or
- Levy fluctuations with parameter $\mu$ (where $p(l) \propto l^{-1-\mu}$ for each move)

### 2. Smoothed Log-Price
The exponentially smoothed last log-price $L(t,k)$ is defined as:

$$L(t,k) = \frac{1}{k} \cdot \text{close}(t) + \frac{k-1}{k} \cdot L(t-1,k)$$

### 3. Agent Price Estimation
Each market participant estimates the fair log-price using their parameters $(\alpha, k, \sigma)$:

$$\text{estimate}(t,\alpha,k,\sigma) = \alpha F(t) + (1-\alpha)L(t,k) + \mathcal{N}(0,\sigma^2)$$

where:
- $\alpha$: insight level about fair price
- $\sigma$: noise level in estimates

### 4. Agent Trading Behavior
#### Order placement
Each agent has parameters:
- Wealth ($W$)
- Delay ($D$)
- Aggressiveness ($A$)
- Uncertainty ($\delta$) in log-space

Orders are placed at prices $\exp(\text{estimate}(t-D,\alpha,k,\sigma) \pm \delta)$ with order size $W \cdot A$
#### Agent pool dynamics
- There is a fixed pool of $N$ agents throughout the simulation
- Each agent:
  - Has persistent state (balance, position, wealth)
  - Tracks their performance relative to initial wealth
  - Gets replaced if wealth drops below 10% of initial
- In each period:
  1. Sample $K \sim \text{Poisson}(\lambda)$ active agents from the pool
  2. Each active agent:
     - Updates their wealth estimate based on current price
     - If wealth < 10% of initial: replaced with new agent
     - Otherwise: generates and submits orders
  3. Process trades and update agent states
- Features:
  - Natural selection of successful strategies
  - Wealth redistribution between agents
  - Position and risk management effects
  - More realistic market dynamics through agent persistence

### 5. Parameter Priors
- $\alpha \sim \text{Beta}(\alpha_0, \alpha_1)$
- $k - 1 \sim \text{Gamma}(k_\alpha, k_\theta)$
- $\sigma \sim \text{Gamma}(\sigma_\alpha, \sigma_\theta)$
- $W \sim \text{Pareto}(w_m, w_\alpha)$
- $D \sim \text{LogNormal}(\mu_D, \sigma_D)$
- $\frac{A}{1-A} \sim \text{LogNormal}(\mu_A, \sigma_A)$
- $\delta \sim \text{Gamma}(\delta_\alpha, \delta_\theta)$

## Exploration Approaches

1. **Direct Sampling**
   - Generate multiple histories with different parameter sets
   - Analyze resulting patterns and statistics

2. **GAN-based Approach**
   - Use Generative Adversarial Networks to match real market data
   - Fine-tune parameters through adversarial training

3. **Differentiable Sequence Model**
   - Train a differentiable model conditioned on parameters
   - Optimize parameter values to maximize fit with real data