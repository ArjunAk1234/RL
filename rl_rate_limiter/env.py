"""
Reinforcement Learning Environment for Risk-Aware Adaptive API Rate Limiting.
Formulated as an MDP:
- State S = (r_t, b_t, c_t, cpu_t, mem_t, l_t, err_t, q_t)
- Action A = {0: Increase, 1: Maintain, 2: Decrease, 3: Throttle, 4: Block}
- Multi-objective Reward R = R_legit - P_abuse - P_overload - P_latency - P_throttle
"""

import numpy as np
from typing import Tuple, Dict, Any, List
from .traffic_generator import TrafficGenerator, ClientRequest
from .backend_simulator import BackendSimulator, BackendMetrics

class RateLimitEnv:
    def __init__(
        self,
        step_duration_sec: float = 1.0,
        episode_length_steps: int = 120,
        target_latency_ms: float = 60.0,
        seed: int = 42
    ):
        self.dt = step_duration_sec
        self.max_steps = episode_length_steps
        self.target_latency = target_latency_ms
        self.current_step = 0
        self.current_time = 0.0

        self.traffic_gen = TrafficGenerator(seed=seed)
        self.backend = BackendSimulator(seed=seed)

        # Rate limiter policy parameters controlled by RL agent:
        # Base capacity multiplier: scale factor on allowable requests per second
        self.current_rate_limit_multiplier = 1.0
        self.high_risk_threshold = 0.65
        self.medium_risk_threshold = 0.35

        # Moving average of request arrival rate for burstiness calculation
        self.history_rates: List[float] = []

        # Current scenario for the episode
        self.scenario = 'mixed'

    def reset(self, scenario: str = 'mixed') -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        self.current_step = 0
        self.current_time = 0.0
        self.scenario = scenario
        self.current_rate_limit_multiplier = 1.0
        self.history_rates.clear()
        self.backend.reset()
        self.traffic_gen.reset()

        initial_obs = self._get_observation(incoming_reqs=[], metrics=self.backend.step(0, self.dt))
        state_idx = self.discretize_state(initial_obs)
        return state_idx, {"raw_obs": initial_obs}

    def _get_observation(self, incoming_reqs: List[ClientRequest], metrics: BackendMetrics) -> Dict[str, float]:
        r_t = len(incoming_reqs) / self.dt
        self.history_rates.append(r_t)
        if len(self.history_rates) > 10:
            self.history_rates.pop(0)

        avg_r = np.mean(self.history_rates) if self.history_rates else 1.0
        b_t = r_t / max(avg_r, 1.0)  # Burstiness

        c_t = float(np.mean([req.risk_score for req in incoming_reqs])) if incoming_reqs else 0.1

        return {
            "r_t": r_t,
            "b_t": b_t,
            "c_t": c_t,
            "cpu_t": metrics.cpu_percent,
            "mem_t": metrics.memory_percent,
            "l_t": metrics.latency_ms,
            "err_t": metrics.error_rate,
            "q_t": float(metrics.queue_length),
            "limit_mult": self.current_rate_limit_multiplier
        }

    def discretize_state(self, obs: Dict[str, float]) -> Tuple[int, ...]:
        """
        Maps continuous 8-dimensional state vector to a compact discrete tuple
        for reliable tabular Q-learning convergence:
        - r_bin (3): Low (<60), Medium (60-180), High (>180)
        - b_bin (2): Normal (<1.3), Bursty (>=1.3)
        - c_bin (3): Low Risk (<0.35), Med Risk (0.35-0.65), High Risk (>0.65)
        - cpu_bin (3): Normal (<65%), Elevated (65-85%), Overload (>=85%)
        - lat_bin (3): Low (<50ms), Medium (50-150ms), High (>150ms)
        Total possible states = 3 * 2 * 3 * 3 * 3 = 162 states.
        """
        r = obs["r_t"]
        r_bin = 0 if r < 60 else (1 if r < 180 else 2)

        b = obs["b_t"]
        b_bin = 0 if b < 1.3 else 1

        c = obs["c_t"]
        c_bin = 0 if c < self.medium_risk_threshold else (1 if c < self.high_risk_threshold else 2)

        cpu = obs["cpu_t"]
        cpu_bin = 0 if cpu < 65.0 else (1 if cpu < 85.0 else 2)

        lat = obs["l_t"]
        lat_bin = 0 if lat < 50.0 else (1 if lat < 150.0 else 2)

        return (r_bin, b_bin, c_bin, cpu_bin, lat_bin)

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, Dict[str, Any]]:
        """
        Action definitions:
        0: Increase rate limit (+25%)
        1: Maintain rate limit
        2: Decrease rate limit (-25%)
        3: Throttle (restrict high and medium risk aggressively)
        4: Block (drop all requests with risk > medium_risk_threshold)
        """
        self.current_step += 1
        self.current_time += self.dt

        # 1. Update policy parameters based on action
        if action == 0:  # Increase
            self.current_rate_limit_multiplier = min(2.5, self.current_rate_limit_multiplier + 0.25)
        elif action == 1:  # Maintain
            pass
        elif action == 2:  # Decrease
            self.current_rate_limit_multiplier = max(0.3, self.current_rate_limit_multiplier - 0.25)
        elif action == 3:  # Throttle
            self.current_rate_limit_multiplier = max(0.2, self.current_rate_limit_multiplier * 0.7)
        elif action == 4:  # Block
            self.current_rate_limit_multiplier = max(0.1, self.current_rate_limit_multiplier * 0.5)

        # 2. Generate incoming traffic for this window
        incoming_reqs = self.traffic_gen.generate_step_traffic(
            current_time=self.current_time,
            dt=self.dt,
            scenario=self.scenario
        )

        # Base nominal quota per client per second
        base_client_quota = int(25 * self.current_rate_limit_multiplier)

        # 3. Apply Admission Control Filter
        admitted: List[ClientRequest] = []
        blocked_legit = 0
        blocked_abuse = 0
        admitted_legit = 0
        admitted_abuse = 0

        client_counts: Dict[str, int] = {}

        for req in incoming_reqs:
            count = client_counts.get(req.client_id, 0)
            
            # Action-specific filtering rules
            is_allowed = True
            if action == 4:  # Block action drops high-risk immediately
                if req.risk_score >= self.medium_risk_threshold:
                    is_allowed = False
                elif count >= base_client_quota:
                    is_allowed = False
            elif action == 3:  # Throttle action tightens quota by risk
                risk_penalty_factor = 1.0 - (req.risk_score * 0.8)
                effective_quota = max(1, int(base_client_quota * risk_penalty_factor))
                if count >= effective_quota:
                    is_allowed = False
            else:
                # Actions 0, 1, 2 use the dynamic multiplier quota
                effective_quota = base_client_quota
                if req.risk_score >= self.high_risk_threshold and action == 2:
                    effective_quota = int(effective_quota * 0.5)
                if count >= effective_quota:
                    is_allowed = False

            if is_allowed:
                admitted.append(req)
                client_counts[req.client_id] = count + 1
                if req.is_malicious:
                    admitted_abuse += 1
                else:
                    admitted_legit += 1
            else:
                if req.is_malicious:
                    blocked_abuse += 1
                else:
                    blocked_legit += 1

        # 4. Backend Processes Admitted Traffic
        metrics = self.backend.step(len(admitted), self.dt)

        # 5. Multi-Objective Reward Calculation
        # R = R_legit - P_abuse - P_overload - P_latency - P_throttle
        r_legit = admitted_legit * 1.0
        p_abuse = admitted_abuse * 3.0
        p_overload = 60.0 if metrics.is_overloaded else 0.0
        p_latency = max(0.0, (metrics.latency_ms - self.target_latency) / 20.0)
        p_throttle = blocked_legit * 3.5  # Heavy penalty for false-positive throttling

        step_reward = r_legit - p_abuse - p_overload - p_latency - p_throttle

        # 6. Next observation and state
        obs = self._get_observation(incoming_reqs, metrics)
        next_state_idx = self.discretize_state(obs)
        done = self.current_step >= self.max_steps

        info = {
            "admitted_legit": admitted_legit,
            "blocked_legit": blocked_legit,
            "admitted_abuse": admitted_abuse,
            "blocked_abuse": blocked_abuse,
            "metrics": metrics,
            "raw_obs": obs,
            "reward_breakdown": {
                "r_legit": r_legit,
                "p_abuse": p_abuse,
                "p_overload": p_overload,
                "p_latency": p_latency,
                "p_throttle": p_throttle
            }
        }

        return next_state_idx, step_reward, done, info
