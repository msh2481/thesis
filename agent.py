from dataclasses import dataclass

import numpy as np
from beartype import beartype as typed
from order_book import Candle, Order


@dataclass
class AgentEstimate:
    timestamp: float
    log_estimate: float


@typed
class Agent:
    def __init__(
        self,
        agent_id: int,
        alpha: float,  # insight level
        k: float,  # smoothing factor
        sigma: float,  # noise level
        wealth: float,  # total wealth
        delay: float,  # trading delay
        aggressiveness: float,  # order size ratio
        uncertainty: float,  # price uncertainty (delta)
    ):
        assert k >= 1, "Smoothing factor must be at least 1"

        self.id = agent_id
        # Agent parameters
        self.alpha = alpha
        self.k = k
        self.sigma = sigma
        self.delay = delay
        self.aggressiveness = aggressiveness
        self.uncertainty = uncertainty

        # Agent state
        self.smoothed_log_price: float | None = None
        self.estimate_history: list[AgentEstimate] = []

        # Financial state
        self.balance = wealth
        self.position = 0.0
        self.wealth = wealth
        self.initial_wealth = wealth

        # Limits
        self.MAX_LOG_PRICE = 10.0
        self.MIN_LOG_PRICE = -10.0

    def update_wealth(self, current_price: float):
        """Update agent's total wealth based on current market price"""
        self.wealth = self.balance + self.position * current_price

    def update_estimate(
        self,
        fair_log_price: float,
        candle: Candle,
        timestamp: float,
        rng: np.random.Generator,
    ) -> float:
        # Update smoothed price using candle close price
        log_close = np.log(candle.close)
        if self.smoothed_log_price is None:
            self.smoothed_log_price = log_close
        else:
            k = self.k
            self.smoothed_log_price = (log_close / k) + (
                self.smoothed_log_price * (k - 1) / k
            )

        # Calculate price estimate
        noise = rng.normal(0, self.sigma)
        log_estimate = (
            self.alpha * fair_log_price
            + (1 - self.alpha) * self.smoothed_log_price
            + noise
        )

        # Store estimate with timestamp
        self.estimate_history.append(AgentEstimate(timestamp, log_estimate))

        # Clean up old estimates (keep last hour for example)
        while len(self.estimate_history) > 3600:  # arbitrary cleanup threshold
            self.estimate_history.pop(0)

        return log_estimate

    def get_delayed_estimate(self, current_time: float) -> float | None:
        """Get the estimate from delay time ago"""
        target_time = current_time - self.delay
        # Find the closest estimate that's not newer than target_time
        valid_estimates = [
            est for est in self.estimate_history if est.timestamp <= target_time
        ]
        if not valid_estimates:
            return None
        return valid_estimates[-1].log_estimate

    def generate_orders(
        self, current_time: float, rng: np.random.Generator
    ) -> tuple[Order, Order] | None:
        """Generate orders using delayed estimate"""
        log_estimate = self.get_delayed_estimate(current_time)
        if log_estimate is None:
            return None

        # Clip log estimate to prevent overflow
        log_estimate = np.clip(log_estimate, self.MIN_LOG_PRICE, self.MAX_LOG_PRICE)

        # Calculate prices with uncertainty
        buy_log_price = np.clip(
            log_estimate - self.uncertainty,
            self.MIN_LOG_PRICE,
            self.MAX_LOG_PRICE,
        )
        sell_log_price = np.clip(
            log_estimate + self.uncertainty,
            self.MIN_LOG_PRICE,
            self.MAX_LOG_PRICE,
        )

        buy_price = np.exp(buy_log_price)
        sell_price = np.exp(sell_log_price)

        # Calculate order size with reasonable limits
        max_buy_size = self.balance / buy_price  # Maximum size we can buy
        max_sell_size = self.position  # Maximum size we can sell
        target_size = self.wealth * self.aggressiveness

        buy_size = min(target_size, max_buy_size)
        sell_size = min(target_size, max_sell_size)

        # Ensure minimum order size
        if max(buy_size, sell_size) < 1e-6:
            return None

        buy_order = Order(
            side="buy",
            price=buy_price,
            size=buy_size,
            timestamp=current_time,
            agent_id=self.id,
            agent=self,  # Add reference to self
        )

        sell_order = Order(
            side="sell",
            price=sell_price,
            size=sell_size,
            timestamp=current_time,
            agent_id=self.id,
            agent=self,  # Add reference to self
        )

        return buy_order, sell_order


@typed
def sample_agent(params_dict: dict, agent_id: int, rng: np.random.Generator) -> Agent:
    """Sample agent with parameters from their prior distributions"""
    alpha = rng.beta(params_dict["alpha_0"], params_dict["alpha_1"])
    k = 1 + rng.gamma(params_dict["k_alpha"], params_dict["k_theta"])
    sigma = rng.gamma(params_dict["sigma_alpha"], params_dict["sigma_theta"])
    wealth = params_dict["w_m"] * rng.pareto(params_dict["w_alpha"])
    proportion = rng.uniform(0.01, 0.99)
    balance = wealth * proportion
    position = wealth * (1 - proportion) / params_dict["initial_price"]
    delay = rng.lognormal(params_dict["mu_D"], params_dict["sigma_D"])

    # Convert A from logit-normal to (0,1)
    logit_A = rng.lognormal(params_dict["mu_A"], params_dict["sigma_A"])
    A = logit_A / (1 + logit_A)

    delta = rng.gamma(params_dict["delta_alpha"], params_dict["delta_theta"])

    agent = Agent(
        agent_id=agent_id,
        alpha=alpha,
        k=k,
        sigma=sigma,
        wealth=wealth,
        delay=delay,
        aggressiveness=A,
        uncertainty=delta,
    )

    # Set initial position and balance
    agent.balance = balance
    agent.position = position

    return agent
