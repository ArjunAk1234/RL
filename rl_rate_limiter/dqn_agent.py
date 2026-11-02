"""
Deep Q-Network (DQN) Agent for Adaptive API Rate Limiting.
Implements:
- 2-layer MLP with ReLU activations using MLX (Apple Silicon-native)
- Experience Replay Buffer with prioritized sampling
- Epsilon-greedy exploration with exponential decay
- Target Network with periodic hard update
- Direct continuous state input (no discretization needed)
"""

import pickle
import numpy as np
from typing import Tuple, Dict, Any, List, Deque
from collections import deque
import os

# ---------------------------------------------------------------------------
# MLX-based Neural Network Components
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular experience replay buffer."""
    def __init__(self, capacity: int = 10000):
        self.buffer: Deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNNetwork:
    """
    Lightweight 2-Layer MLP using MLX:
      Layer 1: input_dim → 128, ReLU
      Layer 2: 128 → 64, ReLU
      Output:  64 → num_actions (Q-values)
    """
    def __init__(self, input_dim: int, num_actions: int, seed: int = 42):
        import mlx.core as mx
        import mlx.nn as nn

        rng = np.random.default_rng(seed)

        # Initialize weights with He (Kaiming) initialization
        def he_weights(in_f, out_f):
            std = np.sqrt(2.0 / in_f)
            return rng.normal(0, std, (in_f, out_f)).astype(np.float32)

        self.W1 = mx.array(he_weights(input_dim, 128))
        self.b1 = mx.zeros((128,))
        self.W2 = mx.array(he_weights(128, 64))
        self.b2 = mx.zeros((64,))
        self.W3 = mx.array(he_weights(64, num_actions))
        self.b3 = mx.zeros((num_actions,))

        # Adam optimizer state
        self.lr = 1e-3
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.t = 0
        self._init_adam()

    def _init_adam(self):
        import mlx.core as mx
        params = self._params()
        self.m = [mx.zeros_like(p) for p in params]
        self.v = [mx.zeros_like(p) for p in params]

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x):
        import mlx.core as mx
        import mlx.nn as nn
        h1 = nn.relu(x @ self.W1 + self.b1)
        h2 = nn.relu(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def predict(self, state_np: np.ndarray) -> np.ndarray:
        import mlx.core as mx
        x = mx.array(state_np.reshape(1, -1))
        out = self.forward(x)
        mx.eval(out)
        return np.array(out).flatten()

    def loss_and_grad(self, states, actions, targets):
        import mlx.core as mx

        def loss_fn(W1, b1, W2, b2, W3, b3):
            import mlx.nn as nn
            h1 = nn.relu(states @ W1 + b1)
            h2 = nn.relu(h1 @ W2 + b2)
            q_all = h2 @ W3 + b3
            # Gather Q-values for chosen actions
            batch_size = q_all.shape[0]
            indices = mx.stack([mx.arange(batch_size), actions], axis=1)
            q_pred = q_all[mx.arange(batch_size), actions]
            loss = mx.mean((q_pred - targets) ** 2)
            return loss

        grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))
        loss, grads = grad_fn(self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
        mx.eval(loss, *grads)
        return float(loss.item()), grads

    def apply_gradients(self, grads):
        import mlx.core as mx
        self.t += 1
        params = self._params()
        names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        new_params = []
        new_m = []
        new_v = []
        for i, (p, g) in enumerate(zip(params, grads)):
            m_new = self.beta1 * self.m[i] + (1 - self.beta1) * g
            v_new = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = m_new / (1 - self.beta1 ** self.t)
            v_hat = v_new / (1 - self.beta2 ** self.t)
            p_new = p - self.lr * m_hat / (mx.sqrt(v_hat) + self.eps)
            new_params.append(p_new)
            new_m.append(m_new)
            new_v.append(v_new)

        mx.eval(*new_params)
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = new_params
        self.m = new_m
        self.v = new_v

    def copy_weights_from(self, other: 'DQNNetwork'):
        """Hard copy weights from another network (target network update)."""
        import mlx.core as mx
        self.W1 = mx.array(np.array(other.W1))
        self.b1 = mx.array(np.array(other.b1))
        self.W2 = mx.array(np.array(other.W2))
        self.b2 = mx.array(np.array(other.b2))
        self.W3 = mx.array(np.array(other.W3))
        self.b3 = mx.array(np.array(other.b3))

    def save(self, path: str):
        import mlx.core as mx
        weights = {
            'W1': np.array(self.W1), 'b1': np.array(self.b1),
            'W2': np.array(self.W2), 'b2': np.array(self.b2),
            'W3': np.array(self.W3), 'b3': np.array(self.b3),
            't': self.t
        }
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, path: str):
        import mlx.core as mx
        with open(path, 'rb') as f:
            w = pickle.load(f)
        self.W1 = mx.array(w['W1'])
        self.b1 = mx.array(w['b1'])
        self.W2 = mx.array(w['W2'])
        self.b2 = mx.array(w['b2'])
        self.W3 = mx.array(w['W3'])
        self.b3 = mx.array(w['b3'])
        self.t = w.get('t', 0)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    DQN Agent operating on continuous 9-dimensional state:
    (r_t, b_t, c_t, cpu_t, mem_t, l_t, err_t, q_t, limit_mult)
    No state discretization needed - neural network is the function approximator.
    """
    STATE_DIM = 9
    
    def __init__(
        self,
        num_actions: int = 5,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
        batch_size: int = 64,
        replay_capacity: int = 10000,
        target_update_freq: int = 10,
        seed: int = 42
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.rng = np.random.default_rng(seed)

        self.online_net = DQNNetwork(self.STATE_DIM, num_actions, seed)
        self.target_net = DQNNetwork(self.STATE_DIM, num_actions, seed)
        self.target_net.copy_weights_from(self.online_net)

        self.replay = ReplayBuffer(replay_capacity)
        self.train_step = 0
        self.losses: List[float] = []

    @staticmethod
    def obs_to_state(obs: Dict[str, float]) -> np.ndarray:
        """Convert raw observation dict to normalized continuous state vector."""
        return np.array([
            obs.get("r_t", 0.0) / 300.0,         # Normalize by max expected RPS
            obs.get("b_t", 1.0) / 3.0,            # Burstiness capped at 3
            obs.get("c_t", 0.0),                   # Already in [0, 1]
            obs.get("cpu_t", 0.0) / 100.0,         # Normalize CPU %
            obs.get("mem_t", 0.0) / 100.0,         # Normalize Memory %
            obs.get("l_t", 0.0) / 2000.0,          # Normalize latency
            obs.get("err_t", 0.0),                  # Already in [0, 1]
            min(obs.get("q_t", 0.0) / 500.0, 1.0), # Queue depth normalized
            obs.get("limit_mult", 1.0) / 2.5       # Policy multiplier normalized
        ], dtype=np.float32)

    def select_action(self, state_vec: np.ndarray, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        q_vals = self.online_net.predict(state_vec)
        return int(np.argmax(q_vals))

    def push(self, state_vec, action, reward, next_state_vec, done):
        self.replay.push(state_vec, action, reward, next_state_vec, done)

    def train_batch(self) -> float:
        if len(self.replay) < self.batch_size:
            return 0.0

        import mlx.core as mx
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        # Compute target Q-values using target network
        next_q = np.array([self.target_net.predict(next_states[i]) for i in range(self.batch_size)])
        next_q_max = np.max(next_q, axis=1)
        targets = rewards + self.gamma * next_q_max * (1.0 - dones)

        # Convert to MLX arrays
        states_mx  = mx.array(states)
        actions_mx = mx.array(actions)
        targets_mx = mx.array(targets.astype(np.float32))

        loss, grads = self.online_net.loss_and_grad(states_mx, actions_mx, targets_mx)
        self.online_net.apply_gradients(grads)
        self.train_step += 1

        # Periodic hard target network update
        if self.train_step % self.target_update_freq == 0:
            self.target_net.copy_weights_from(self.online_net)

        self.losses.append(loss)
        return loss

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.online_net.save(path)
        with open(path.replace('.pkl', '_meta.pkl'), 'wb') as f:
            pickle.dump({'epsilon': self.epsilon, 'train_step': self.train_step}, f)

    def load(self, path: str):
        self.online_net.load(path)
        self.target_net.copy_weights_from(self.online_net)
        meta_path = path.replace('.pkl', '_meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.epsilon = meta.get('epsilon', self.epsilon_min)
            self.train_step = meta.get('train_step', 0)

    @property
    def action_names(self) -> Dict[int, str]:
        return {
            0: "Increase (+25%)",
            1: "Maintain",
            2: "Decrease (-25%)",
            3: "Throttle (Risk-Aware)",
            4: "Block High-Risk"
        }
