import numpy as np
from agent import Agent, Candle, Order, sample_agent, Trade
from beartype import beartype as typed
from loguru import logger
from numpy.random import default_rng
from order_book import OrderBook

# Configure numpy printing format
np.set_printoptions(
    precision=2, suppress=True, formatter={"float": lambda x: f"{x:.2f}"}
)


@typed
def generate_fair_price(
    steps: int,
    initial_price: float,
    scale: float,
    rng: np.random.Generator,
    levy: bool = False,
    levy_param: float | None = None,
) -> list[float]:
    """Generate fair price process"""
    MAX_LOG_PRICE = 10.0
    MIN_LOG_PRICE = -10.0

    log_price = np.clip(np.log(initial_price), MIN_LOG_PRICE, MAX_LOG_PRICE)
    log_prices = [log_price]

    for _ in range(steps - 1):
        if levy:
            assert levy_param is not None
            # Simplified Levy process using power law
            step = rng.standard_cauchy() * scale
            # Truncate extreme values
            step = np.clip(step, -5 * scale, 5 * scale)
        else:
            step = rng.normal(0, scale)

        log_price = np.clip(log_price + step, MIN_LOG_PRICE, MAX_LOG_PRICE)
        log_prices.append(log_price)

    return [np.exp(p) for p in log_prices]


@typed
def simulate(
    params: dict,
    steps: int,
    n_agents: int = 100,
    seed: int | None = None,
) -> tuple[list[Trade], list[float], list[Agent]]:
    """Implementation of the persistent process with fixed agent pool"""
    rng = default_rng(seed)
    book = OrderBook()
    trades: list[Trade] = []
    candles: list[Candle] = []

    # Generate fair price process
    fair_prices = generate_fair_price(
        steps=steps,
        initial_price=params["initial_price"],
        scale=params["price_scale"],
        rng=rng,
        levy=params.get("use_levy", False),
        levy_param=params.get("levy_param"),
    )

    current_price = params["initial_price"]
    next_agent_id = 0

    # Initialize agent pool
    agents = []
    for _ in range(n_agents):
        agent = sample_agent(params, next_agent_id, rng)
        # logger.info(f"Initialized agent {next_agent_id} with wealth {agent.wealth}")
        agents.append(agent)
        next_agent_id += 1

    # Simulate trading process
    for t in range(steps):
        # logger.info(f"Simulating period {t}. Current price: {current_price}")
        period_trades: list[Trade] = []

        # Taxes to ensure that all agents have positive wealth
        tax_rate = 0.01
        collected_taxes = 0.0
        for agent in agents:
            amount = agent.wealth * tax_rate
            agent.balance -= amount
            agent.wealth -= amount
            collected_taxes += amount
        for agent in agents:
            amount = collected_taxes / len(agents)
            agent.balance += amount
            agent.wealth += amount

        # Create candle for previous period if there were trades
        if t > 0:
            candle = Candle.from_trades(period_trades, float(t - 1))
            if candle is None:  # No trades in period, create synthetic candle
                last_price = current_price
                candle = Candle(
                    timestamp=float(t - 1),
                    open=last_price,
                    high=last_price,
                    low=last_price,
                    close=last_price,
                    volume=0.0,
                    num_trades=0,
                )
            candles.append(candle)

        # Sample active agents for this period
        n_active = rng.poisson(params["lambda"])
        active_agents = rng.choice(
            agents, size=min(n_active, len(agents)), replace=False
        )

        if t % 500 == 0:
            logger.debug(f"Saving agent status at t={t}")
            logger.info(
                f"Order book size: {sum(len(h) for h in book.heaps.values())} | Current price: {current_price:.2f}"
            )
            with open(f"logs/agents_{t}.txt", "w") as f:
                print(
                    f"\nAgent status at t={t}:\n",
                    file=f,
                )
                print(
                    f"{'ID':>4} {'Alpha/Sigma':>8} {'Uncert':>8} {'Position':>12} {'Balance':>12} {'Wealth':>12} {'Wealth Δ':>10}",
                    file=f,
                )
                print("-" * 85, file=f)
                for agent in sorted(agents, key=lambda x: x.wealth, reverse=True):
                    wealth_change = agent.wealth - agent.initial_wealth
                    print(
                        f"{agent.id:4d} {agent.alpha/max(agent.sigma, 1e-3):8.3f} {agent.uncertainty:8.3f} "
                        f"{agent.position:12.3f} {agent.balance:12.2f} {agent.wealth:12.2f} {wealth_change:10.2f}",
                        file=f,
                    )

        for i, agent in enumerate(active_agents):
            # Update agent's wealth estimate
            agent.update_wealth(current_price)

            if candles:
                _ = agent.update_estimate(
                    np.log(fair_prices[t]), candles[-1], float(t), rng
                )

            # Generate and submit orders with current time
            orders = agent.generate_orders(float(t), rng)
            if orders is not None:
                buy_order, sell_order = orders
                # logger.info(
                #     f"Agent {agent.id} (position={agent.position}; balance={agent.balance}; alpha={agent.alpha}) submitted orders:\n"
                #     f"  {buy_order}\n"
                #     f"  {sell_order}"
                # )
                # Process orders
                for order in [buy_order, sell_order]:
                    new_trades = book.add_order(order)
                    for trade in new_trades:
                        # logger.info(f"New trade: {trade}")
                        trades.append(trade)
                        period_trades.append(trade)
                        current_price = trade.price

        # logger.info(f"Period {t} ends with order book:\n{book}")

    return trades, fair_prices, agents
