"""
Benchmarking Suite: Evaluates RL Agent against 4 Conventional Baselines
across multiple realistic traffic scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .env import RateLimitEnv
from .agent import QLearningAgent
from .baselines import (
    TokenBucketLimiter,
    LeakyBucketLimiter,
    FixedWindowLimiter,
    SlidingWindowLimiter,
    BaseRateLimiter
)
from .backend_simulator import BackendSimulator

class BaselineRunner:
    """
    Simulates standard rate limiting where fixed algorithms control admission.
    """
    def __init__(self, limiter: BaseRateLimiter, name: str):
        self.limiter = limiter
        self.name = name

    def run_episode(
        self,
        env: RateLimitEnv,
        scenario: str = 'mixed'
    ) -> Dict[str, Any]:
        self.limiter.reset()
        backend = BackendSimulator(seed=42)
        traffic_gen = env.traffic_gen

        admitted_legit = 0
        blocked_legit = 0
        admitted_abuse = 0
        blocked_abuse = 0
        overload_events = 0
        latencies = []
        cpus = []
        cumulative_reward = 0.0

        for step in range(env.max_steps):
            t_curr = step * env.dt
            reqs = traffic_gen.generate_step_traffic(t_curr, env.dt, scenario)
            
            admitted_step = 0
            for req in reqs:
                allowed = self.limiter.allow_request(req.client_id, req.timestamp)
                if allowed:
                    admitted_step += 1
                    if req.is_malicious:
                        admitted_abuse += 1
                    else:
                        admitted_legit += 1
                else:
                    if req.is_malicious:
                        blocked_abuse += 1
                    else:
                        blocked_legit += 1

            metrics = backend.step(admitted_step, env.dt)
            latencies.append(metrics.latency_ms)
            cpus.append(metrics.cpu_percent)
            if metrics.is_overloaded:
                overload_events += 1

            # Calculate equivalent reward for comparison
            r_legit = admitted_legit * 1.0
            p_abuse = admitted_abuse * 3.0
            p_overload = 60.0 if metrics.is_overloaded else 0.0
            p_lat = max(0.0, (metrics.latency_ms - env.target_latency) / 20.0)
            p_throttle = blocked_legit * 3.5
            cumulative_reward += (r_legit - p_abuse - p_overload - p_lat - p_throttle)

        total_legit = admitted_legit + blocked_legit
        total_abuse = admitted_abuse + blocked_abuse

        return {
            "Method": self.name,
            "Scenario": scenario,
            "Legit Acceptance (%)": round(admitted_legit / max(1, total_legit) * 100, 2),
            "Abuse Blocked (%)": round(blocked_abuse / max(1, total_abuse) * 100, 2),
            "Avg Latency (ms)": round(float(np.mean(latencies)), 2),
            "Avg CPU (%)": round(float(np.mean(cpus)), 2),
            "Overload Events": overload_events,
            "False Throttling (%)": round(blocked_legit / max(1, total_legit) * 100, 2),
            "Cumulative Reward": round(cumulative_reward, 1)
        }


def evaluate_rl_agent(
    env: RateLimitEnv,
    agent: QLearningAgent,
    scenario: str = 'mixed'
) -> Dict[str, Any]:
    state, info = env.reset(scenario)
    
    admitted_legit = 0
    blocked_legit = 0
    admitted_abuse = 0
    blocked_abuse = 0
    overload_events = 0
    latencies = []
    cpus = []
    cumulative_reward = 0.0

    done = False
    while not done:
        action = agent.select_action(state, training=False)
        next_state, reward, done, step_info = env.step(action)

        admitted_legit += step_info["admitted_legit"]
        blocked_legit += step_info["blocked_legit"]
        admitted_abuse += step_info["admitted_abuse"]
        blocked_abuse += step_info["blocked_abuse"]
        
        metrics = step_info["metrics"]
        latencies.append(metrics.latency_ms)
        cpus.append(metrics.cpu_percent)
        if metrics.is_overloaded:
            overload_events += 1

        cumulative_reward += reward
        state = next_state

    total_legit = admitted_legit + blocked_legit
    total_abuse = admitted_abuse + blocked_abuse

    return {
        "Method": "Adaptive RL (Q-Learning)",
        "Scenario": scenario,
        "Legit Acceptance (%)": round(admitted_legit / max(1, total_legit) * 100, 2),
        "Abuse Blocked (%)": round(blocked_abuse / max(1, total_abuse) * 100, 2),
        "Avg Latency (ms)": round(float(np.mean(latencies)), 2),
        "Avg CPU (%)": round(float(np.mean(cpus)), 2),
        "Overload Events": overload_events,
        "False Throttling (%)": round(blocked_legit / max(1, total_legit) * 100, 2),
        "Cumulative Reward": round(cumulative_reward, 1)
    }


def run_full_benchmark(
    env: RateLimitEnv,
    agent: QLearningAgent,
    scenarios: List[str] = None
) -> pd.DataFrame:
    if scenarios is None:
        scenarios = ['normal', 'burst', 'abuse', 'overload', 'mixed']

    baselines = [
        BaselineRunner(TokenBucketLimiter(capacity=35, refill_rate=25), "Token Bucket"),
        BaselineRunner(LeakyBucketLimiter(capacity=35, leak_rate=25), "Leaky Bucket"),
        BaselineRunner(FixedWindowLimiter(max_requests=25, window_size=1.0), "Fixed Window"),
        BaselineRunner(SlidingWindowLimiter(max_requests=25, window_size=1.0), "Sliding Window")
    ]

    all_results = []
    for sc in scenarios:
        for b in baselines:
            res = b.run_episode(env, sc)
            all_results.append(res)
        rl_res = evaluate_rl_agent(env, agent, sc)
        all_results.append(rl_res)

    df = pd.DataFrame(all_results)
    return df
