import gymnasium as gym
import numpy as np
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from beartype.typing import TypeAlias
from jaxtyping import Float, Int
from loguru import logger
from numpy import ndarray as ND

# Type aliases for clarity
State: TypeAlias = Float[ND, "1"]
Action: TypeAlias = Float[ND, "1"]


class DebugEnv(gym.Env):
    """
    A simple environment for debugging RL agents.
    State space: 1D float in [-1, 1]
    Action space: 1D float in [-1, 1]
    Reward: -MSE between previous state and current action
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.state: State = np.zeros(1, dtype=np.float32)
        self._steps = 0
        self.max_steps = 100  # Episode length

    @typed
    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[State, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Initialize state randomly in [-1, 1]
        self.state = self.np_random.uniform(low=-1.0, high=1.0, size=(1,)).astype(
            np.float32
        )
        self._steps = 0

        return self.state, {}

    @typed
    def step(self, action: Action) -> tuple[State, float, bool, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Float value in [-1, 1]

        Returns:
            (next_state, reward, terminated, truncated, info)
        """
        assert_type(action, Action)
        self._steps += 1

        # Calculate reward as negative MSE between previous state and action
        reward = -float(np.mean((self.state - action) ** 2))

        # Generate new state
        self.state = self.np_random.uniform(low=-1.0, high=1.0, size=(1,)).astype(
            np.float32
        )

        # Check if episode should end
        terminated = False
        truncated = self._steps >= self.max_steps

        return self.state, reward, terminated, truncated, {}

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            logger.info(f"State: {self.state[0]:.3f}")

    def close(self):
        """Clean up resources."""
        pass
