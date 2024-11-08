import numpy as np
from agent import sample_agent
from beartype import beartype as typed
from loguru import logger
from numpy.random import default_rng
from params import some_insight_params
from price_process import simulate
from visualization import visualize_agents, visualize_trades

# Configure numpy printing format
np.set_printoptions(
    precision=2, suppress=True, formatter={"float": lambda x: f"{x:.2f}"}
)

logger.remove()
with open("logs/market_simulation.log", "w") as f:
    f.flush()
logger.add("logs/market_simulation.log")

if __name__ == "__main__":
    params = some_insight_params
    print("Sampling initial agents...")
    seed = 42
    rng = default_rng(seed)
    initial_agents = [sample_agent(params, i, rng) for i in range(1000)]
    visualize_agents(
        initial_agents,
        title="Initial Agent Population",
        filename="img/agents_initial.png",
    )

    print("Running simulation...")
    trades, fair_prices, final_agents = simulate(
        params=params,
        steps=200,
        seed=seed,
        n_agents=10,
    )

    print("Visualizing results...")
    visualize_trades(trades, fair_prices, use_candles=False, filename="img/trades.png")
    visualize_trades(trades, fair_prices, use_candles=True, filename="img/candles.png")

    print("Analyzing final agent pool...")
    visualize_agents(
        final_agents, title="Final Agent Population", filename="img/agents_final.png"
    )
