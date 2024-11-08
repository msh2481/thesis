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
        logger.info(f"Initialized agent {next_agent_id} with wealth {agent.wealth}")
        agents.append(agent)
        next_agent_id += 1

    # Simulate trading process
    for t in range(steps):
        logger.info(f"Simulating period {t}. Current price: {current_price}")
        period_trades: list[Trade] = []

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
            logger.info(f"Created candle {candle}")
            candles.append(candle)

        # Sample active agents for this period
        n_active = rng.poisson(params["lambda"])
        active_agents = rng.choice(
            agents, size=min(n_active, len(agents)), replace=False
        )

        if t % 100 == 0:
            logger.debug(f"Saving agent status at t={t}")
            with open(f"logs/agents_{t}.txt", "w") as f:
                print("\nAgent status at t=100:\n", file=f)
                print(
                    f"{'ID':>4} {'Alpha':>8} {'Position':>12} {'Balance':>12} {'Wealth':>12} {'Wealth Δ%':>10}",
                    file=f,
                )
                print("-" * 65, file=f)
                for agent in sorted(agents, key=lambda x: x.wealth, reverse=True):
                    wealth_change = (agent.wealth / agent.initial_wealth - 1) * 100
                    print(
                        f"{agent.id:4d} {agent.alpha:8.3f} {agent.position:12.3f} "
                        f"{agent.balance:12.2f} {agent.wealth:12.2f} {wealth_change:10.1f}%",
                        file=f,
                    )

        for i, agent in enumerate(active_agents):
            # Update agent's wealth estimate
            agent.update_wealth(current_price)

            # Check if agent needs replacement
            if agent.wealth < 0.1 * agent.initial_wealth:
                logger.info(
                    f"Agent {agent.id} (position={agent.position}; balance={agent.balance}; alpha={agent.alpha}) goes bankrupt"
                )
                # Create new agent with same ID but new parameters
                new_agent = sample_agent(params, agent.id, rng)
                agents[i] = new_agent
                logger.info(f"Replaced with new agent: {new_agent}")
                continue

            if candles:
                _ = agent.update_estimate(
                    np.log(fair_prices[t]), candles[-1], float(t), rng
                )

            # Generate and submit orders with current time
            orders = agent.generate_orders(float(t), rng)
            if orders is not None:
                buy_order, sell_order = orders
                logger.info(
                    f"Agent {agent.id} (position={agent.position}; balance={agent.balance}; alpha={agent.alpha}) submitted orders:\n"
                    f"  {buy_order}\n"
                    f"  {sell_order}"
                )
                # Process orders
                for order in [buy_order, sell_order]:
                    new_trades = book.add_order(order)
                    for trade in new_trades:
                        logger.info(f"New trade: {trade}")
                        trades.append(trade)
                        period_trades.append(trade)
                        current_price = trade.price

        logger.info(f"Period {t} ends with order book:\n{book}")

    return trades, fair_prices, agents
