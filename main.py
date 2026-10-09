"""
Main Driver Script for Risk-Aware Adaptive API Rate Limiting Project.
Executes:
1. Q-Learning Agent Training & Convergence Loop
2. Model Serialization (models/q_table.pkl)
3. Multi-Scenario Benchmarking against Token Bucket, Leaky Bucket, Fixed & Sliding Window
4. Dynamic Stress Timeline Simulation
5. Generation of Publication-Quality Plots & CSV Reports
"""

import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate

from rl_rate_limiter.env import RateLimitEnv
from rl_rate_limiter.agent import QLearningAgent
from rl_rate_limiter.benchmark import run_full_benchmark, BaselineRunner
from rl_rate_limiter.baselines import TokenBucketLimiter
from rl_rate_limiter.backend_simulator import BackendSimulator
from rl_rate_limiter.visualize import (
    plot_training_curve,
    plot_dynamic_timeline,
    plot_benchmark_bars
)

def train_agent(
    env: RateLimitEnv,
    num_episodes: int = 180,
    save_path: str = "models/q_table.pkl"
) -> Tuple[QLearningAgent, List[float], List[float]]:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent = QLearningAgent(
        num_actions=5,
        alpha=0.15,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.97
    )

    scenarios = ['normal', 'burst', 'abuse', 'mixed', 'overload']
    episode_rewards = []
    epsilons = []

    print(f"\n[Training] Starting Q-learning training for {num_episodes} episodes...")
    for ep in range(1, num_episodes + 1):
        # Alternate scenarios during training to ensure diverse exploration
        sc = scenarios[ep % len(scenarios)]
        state, _ = env.reset(scenario=sc)
        total_ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_ep_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_ep_reward)
        epsilons.append(agent.epsilon)

        if ep % 30 == 0 or ep == num_episodes:
            avg_rew = np.mean(episode_rewards[-20:])
            print(f"  Episode {ep:3d}/{num_episodes} | Scenario: {sc:8s} | "
                  f"Avg Reward (last 20): {avg_rew:7.1f} | Epsilon: {agent.epsilon:.3f} | States Explored: {len(agent.q_table)}")

    agent.save(save_path)
    print(f"[Model Saved] Policy serialized to {save_path} ({len(agent.q_table)} state entries).")
    return agent, episode_rewards, epsilons


def run_dynamic_timeline_comparison(env: RateLimitEnv, agent: QLearningAgent):
    """
    Executes a fine-grained 60-step run comparing Adaptive RL against Token Bucket
    under a mixed surge scenario.
    """
    steps = 60
    # 1. RL Run
    env.max_steps = steps
    state, _ = env.reset(scenario='mixed')
    rl_data = {"cpu": [], "latency": [], "admitted_legit": [], "blocked_abuse": []}
    for _ in range(steps):
        action = agent.select_action(state, training=False)
        state, _, _, info = env.step(action)
        rl_data["cpu"].append(info["metrics"].cpu_percent)
        rl_data["latency"].append(info["metrics"].latency_ms)
        rl_data["admitted_legit"].append(info["admitted_legit"])
        rl_data["blocked_abuse"].append(info["blocked_abuse"])

    # 2. Token Bucket Run
    tb = TokenBucketLimiter(capacity=35, refill_rate=25)
    tb_backend = BackendSimulator(seed=42)
    tb_data = {"cpu": [], "latency": [], "admitted_legit": [], "blocked_abuse": []}
    for step in range(steps):
        t_curr = step * env.dt
        reqs = env.traffic_gen.generate_step_traffic(t_curr, env.dt, 'mixed')
        admitted = 0
        legit_adm = 0
        abuse_blk = 0
        for req in reqs:
            if tb.allow_request(req.client_id, req.timestamp):
                admitted += 1
                if not req.is_malicious:
                    legit_adm += 1
            else:
                if req.is_malicious:
                    abuse_blk += 1
        metrics = tb_backend.step(admitted, env.dt)
        tb_data["cpu"].append(metrics.cpu_percent)
        tb_data["latency"].append(metrics.latency_ms)
        tb_data["admitted_legit"].append(legit_adm)
        tb_data["blocked_abuse"].append(abuse_blk)

    plot_dynamic_timeline(rl_data, tb_data)


def main():
    print("=" * 80)
    print("  RISK-AWARE ADAPTIVE API RATE LIMITING USING REINFORCEMENT LEARNING")
    print("  B. Tech CSE (AI) - 22AIE401 Course Project")
    print("=" * 80)

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Initialize RL Environment
    env = RateLimitEnv(step_duration_sec=1.0, episode_length_steps=80, seed=42)

    # 2. Train Q-Learning Agent
    agent, rewards, epsilons = train_agent(env, num_episodes=180)

    # 3. Generate Training Convergence Plot
    plot_training_curve(rewards, epsilons)

    # 4. Run Benchmark Suite across All Scenarios
    print("\n[Benchmarking] Evaluating RL Agent vs Baselines across all traffic profiles...")
    benchmark_df = run_full_benchmark(env, agent)
    benchmark_df.to_csv("results/benchmark_results.csv", index=False)
    print("[Report Saved] Full benchmark metrics saved to results/benchmark_results.csv")

    # Display Pretty Markdown/Ascii Table
    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS SUMMARY (Averaged across test scenarios)")
    print("=" * 80)
    summary_df = benchmark_df.groupby('Method')[
        ['Legit Acceptance (%)', 'Abuse Blocked (%)', 'Avg Latency (ms)', 'Avg CPU (%)', 'Overload Events', 'False Throttling (%)', 'Cumulative Reward']
    ].mean().round(2).reset_index()

    print(tabulate(summary_df, headers='keys', tablefmt='github', showindex=False))

    # 5. Generate Benchmark Plots
    plot_benchmark_bars(benchmark_df)

    # 6. Run Dynamic Timeline Simulation
    print("\n[Simulation] Generating dynamic stress timeline comparison...")
    run_dynamic_timeline_comparison(env, agent)

    print("\n" + "=" * 80)
    print("  EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("  Artifacts generated:")
    print("    - models/q_table.pkl")
    print("    - results/benchmark_results.csv")
    print("    - results/training_curve.png")
    print("    - results/dynamic_stress_timeline.png")
    print("    - results/benchmark_comparison.png")
    print("=" * 80)

if __name__ == "__main__":
    from typing import Tuple, List
    main()
