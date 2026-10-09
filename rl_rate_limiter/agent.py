"""
Q-Learning Agent for Adaptive API Rate Limiting.
Implements:
- Tabular Q-learning with Bellman optimality update
- Epsilon-greedy exploration with exponential decay
- Policy extraction and Q-table persistence (save/load)
"""

import pickle
import numpy as np
from typing import Tuple, Dict, Any, Optional

class QLearningAgent:
    def __init__(
        self,
        num_actions: int = 5,
        alpha: float = 0.15,        # Learning rate
        gamma: float = 0.95,        # Discount factor
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42
    ):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Q-table: maps discrete state tuple -> np.ndarray of shape (num_actions,)
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def get_q_values(self, state: Tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            # Initialize optimistically or at zero
            self.q_table[state] = np.zeros(self.num_actions, dtype=np.float64)
        return self.q_table[state]

    def select_action(self, state: Tuple[int, ...], training: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        If training=False, pure exploitation (argmax).
        """
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))

        q_vals = self.get_q_values(state)
        # Random tie-breaking for equal max Q-values
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool
    ) -> float:
        """
        Bellman equation update:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        q_current = self.get_q_values(state)[action]
        
        if done:
            target = reward
        else:
            q_next_max = np.max(self.get_q_values(next_state))
            target = reward + self.gamma * q_next_max

        td_error = target - q_current
        self.q_table[state][action] += self.alpha * td_error

        return float(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "gamma": self.gamma
            }, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data["q_table"]
            self.epsilon = data.get("epsilon", self.epsilon_min)
            self.alpha = data.get("alpha", self.alpha)
            self.gamma = data.get("gamma", self.gamma)

    @property
    def action_names(self) -> Dict[int, str]:
        return {
            0: "Increase (+25%)",
            1: "Maintain",
            2: "Decrease (-25%)",
            3: "Throttle (Risk-Aware)",
            4: "Block High-Risk"
        }
