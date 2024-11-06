from agent import sample_agent
from beartype import beartype as typed
from numpy.random import default_rng
from params import some_insight_params
from price_process import simulate
from visualization import visualize_agents, visualize_trades


if __name__ == "__main__":
    params = some_insight_params
    print("Sampling initial agents...")
    rng = default_rng(42)
    initial_agents = [sample_agent(params, i, rng) for i in range(1000)]
    visualize_agents(
        initial_agents,
        title="Initial Agent Population",
        filename="img/agents_initial.png",
    )

    print("Running simulation...")
    trades, fair_prices, final_agents = simulate(
        params=params,
        steps=500,
        # seed=40,
        n_agents=100,
    )

    print("Visualizing results...")
    visualize_trades(trades, fair_prices, use_candles=False, filename="img/trades.png")
    visualize_trades(trades, fair_prices, use_candles=True, filename="img/candles.png")

    print("Analyzing final agent pool...")
    visualize_agents(
        final_agents, title="Final Agent Population", filename="img/agents_final.png"
    )
