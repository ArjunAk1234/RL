"""
DQN Training Driver — Side-by-side comparison with Tabular Q-Learning.
Runs DQN training and generates a comparison visualisation and CSV.
"""

import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate

from rl_rate_limiter.env import RateLimitEnv
from rl_rate_limiter.dqn_agent import DQNAgent
from rl_rate_limiter.benchmark import run_full_benchmark, evaluate_rl_agent
from rl_rate_limiter.agent import QLearningAgent
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_dqn(
    num_episodes: int = 180,
    save_path: str = "models/dqn_weights.pkl"
) -> tuple:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = RateLimitEnv(step_duration_sec=1.0, episode_length_steps=80, seed=42)
    agent = DQNAgent(
        num_actions=5,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.97,
        batch_size=64,
        replay_capacity=10000,
        target_update_freq=10,
        seed=42
    )

    scenarios = ['normal', 'burst', 'abuse', 'mixed', 'overload']
    episode_rewards = []
    epsilons = []

    print(f"\n[DQN Training] Starting MLX-based DQN training for {num_episodes} episodes...")
    for ep in range(1, num_episodes + 1):
        sc = scenarios[ep % len(scenarios)]
        state, info = env.reset(scenario=sc)
        obs = info["raw_obs"]
        state_vec = DQNAgent.obs_to_state(obs)

        total_ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state_vec, training=True)
            next_state_idx, reward, done, step_info = env.step(action)
            next_obs = step_info["raw_obs"]
            next_vec = DQNAgent.obs_to_state(next_obs)

            agent.push(state_vec, action, reward, next_vec, done)
            agent.train_batch()

            state_vec = next_vec
            total_ep_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_ep_reward)
        epsilons.append(agent.epsilon)

        if ep % 30 == 0 or ep == num_episodes:
            avg_rew = np.mean(episode_rewards[-20:])
            loss = agent.losses[-1] if agent.losses else 0.0
            print(f"  Episode {ep:3d}/{num_episodes} | Scenario: {sc:8s} | "
                  f"Avg Reward (last 20): {avg_rew:8.1f} | ε: {agent.epsilon:.3f} | Loss: {loss:.4f}")

    agent.save(save_path)
    print(f"[DQN Model Saved] Weights saved to {save_path}")
    return agent, episode_rewards, epsilons, env


def evaluate_dqn_agent(env: RateLimitEnv, agent: DQNAgent, scenario: str) -> dict:
    """Evaluate the trained DQN agent on a scenario (mirrors benchmark.evaluate_rl_agent)."""
    state, info = env.reset(scenario)
    obs = info["raw_obs"]
    state_vec = DQNAgent.obs_to_state(obs)

    admitted_legit = 0; blocked_legit = 0
    admitted_abuse = 0; blocked_abuse = 0
    overload_events = 0; latencies = []; cpus = []
    cumulative_reward = 0.0
    done = False

    while not done:
        action = agent.select_action(state_vec, training=False)
        _, reward, done, info_ = env.step(action)
        obs = info_["raw_obs"]
        state_vec = DQNAgent.obs_to_state(obs)

        admitted_legit += info_["admitted_legit"]
        blocked_legit  += info_["blocked_legit"]
        admitted_abuse += info_["admitted_abuse"]
        blocked_abuse  += info_["blocked_abuse"]
        m = info_["metrics"]
        latencies.append(m.latency_ms)
        cpus.append(m.cpu_percent)
        if m.is_overloaded: overload_events += 1
        cumulative_reward += reward

    total_legit = admitted_legit + blocked_legit
    total_abuse = admitted_abuse + blocked_abuse
    return {
        "Method":               "DQN (MLX 2-Layer MLP)",
        "Scenario":             scenario,
        "Legit Acceptance (%)": round(admitted_legit / max(1, total_legit) * 100, 2),
        "Abuse Blocked (%)":    round(blocked_abuse  / max(1, total_abuse)  * 100, 2),
        "Avg Latency (ms)":     round(float(np.mean(latencies)), 2),
        "Avg CPU (%)":          round(float(np.mean(cpus)), 2),
        "Overload Events":      overload_events,
        "False Throttling (%)": round(blocked_legit  / max(1, total_legit) * 100, 2),
        "Cumulative Reward":    round(cumulative_reward, 1)
    }


def plot_dqn_vs_tabular(
    dqn_rewards: list, dqn_epsilons: list,
    tab_rewards: list, tab_epsilons: list,
    save_path: str = "results/dqn_vs_tabular_training.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    window = max(5, len(dqn_rewards) // 20)
    roll_dqn = pd.Series(dqn_rewards).rolling(window, min_periods=1).mean()
    roll_tab = pd.Series(tab_rewards).rolling(window, min_periods=1).mean()

    ax = axes[0]
    ax.plot(dqn_rewards, color='#f28e2b', alpha=0.25, linewidth=0.8)
    ax.plot(roll_dqn,    color='#f28e2b', linewidth=2.2, label='DQN (MLX)')
    ax.plot(tab_rewards, color='#4e79a7', alpha=0.25, linewidth=0.8)
    ax.plot(roll_tab,    color='#4e79a7', linewidth=2.2, label='Tabular Q-Learning')
    ax.set_xlabel('Episode', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontsize=10, fontweight='bold')
    ax.set_title('Training Convergence: DQN vs Tabular Q-Learning', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)

    ax2 = axes[1]
    ax2.plot(dqn_epsilons, color='#f28e2b', linewidth=2, label='DQN ε')
    ax2.plot(tab_epsilons, color='#4e79a7', linewidth=2, label='Tabular ε')
    ax2.set_xlabel('Episode', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Epsilon (ε)', fontsize=10, fontweight='bold')
    ax2.set_title('Exploration Decay Comparison', fontsize=12, fontweight='bold')
    ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] DQN vs Tabular training comparison: {save_path}")


def plot_head_to_head(combined_df: pd.DataFrame, save_path: str = "results/dqn_benchmark_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary = combined_df.groupby('Method')[
        ['Legit Acceptance (%)', 'Abuse Blocked (%)', 'Avg Latency (ms)', 'Overload Events']
    ].mean().reset_index()

    palette = {
        'Adaptive RL (Q-Learning)': '#4e79a7',
        'DQN (MLX 2-Layer MLP)':   '#f28e2b',
        'Token Bucket':             '#e15759',
        'Leaky Bucket':             '#76b7b2',
        'Fixed Window':             '#59a14f',
        'Sliding Window':           '#b07aa1'
    }
    colors = [palette.get(m, '#aaaaaa') for m in summary['Method']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics = [
        ('Legit Acceptance (%)', 'Legitimate Request Acceptance (%)', axes[0, 0]),
        ('Abuse Blocked (%)',    'Abuse / Bot Blocking Rate (%)',      axes[0, 1]),
        ('Avg Latency (ms)',     'Average Response Latency (ms)',      axes[1, 0]),
        ('Overload Events',      'Total Backend Overload Incidents',   axes[1, 1])
    ]

    for col, title, ax in metrics:
        bars = ax.bar(summary['Method'], summary[col], color=colors, edgecolor='black', alpha=0.88, width=0.55)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_xticks(range(len(summary['Method'])))
        ax.set_xticklabels(summary['Method'], rotation=18, ha='right', fontsize=8)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    plt.suptitle('Full Evaluation: DQN vs Tabular Q-Learning vs Industry Baselines',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot Saved] DQN head-to-head comparison: {save_path}")


if __name__ == "__main__":
    # ---- 1. Train DQN ----
    dqn_agent, dqn_rewards, dqn_epsilons, dqn_env = train_dqn(num_episodes=180)

    # ---- 2. Load pre-trained Tabular Q agent ----
    tab_env = RateLimitEnv(step_duration_sec=1.0, episode_length_steps=80, seed=42)
    tab_agent = QLearningAgent()
    if os.path.exists("models/q_table.pkl"):
        tab_agent.load("models/q_table.pkl")
        print("[Tabular] Loaded pre-trained Q-table from models/q_table.pkl")
    else:
        print("[Tabular] No pre-trained Q-table found. Run main.py first.")
        sys.exit(1)

    # ---- 3. Reload tabular training curve from saved metadata if available ----
    tab_rewards_path = "models/tab_training_history.npy"
    if os.path.exists(tab_rewards_path):
        saved = np.load(tab_rewards_path)
        tab_rewards   = list(saved[0])
        tab_epsilons  = list(saved[1])
    else:
        tab_rewards  = [0] * 180
        tab_epsilons = [0.05] * 180

    # ---- 4. Training Curve Comparison Plot ----
    plot_dqn_vs_tabular(dqn_rewards, dqn_epsilons, tab_rewards, tab_epsilons)

    # ---- 5. Benchmark DQN across all scenarios ----
    print("\n[DQN Benchmarking] Evaluating DQN across all traffic profiles...")
    scenarios = ['normal', 'burst', 'abuse', 'overload', 'mixed']
    dqn_results = [evaluate_dqn_agent(dqn_env, dqn_agent, sc) for sc in scenarios]

    # ---- 6. Load existing baseline + tabular results ----
    if os.path.exists("results/benchmark_results.csv"):
        existing_df = pd.read_csv("results/benchmark_results.csv")
    else:
        existing_df = run_full_benchmark(tab_env, tab_agent)

    combined_df = pd.concat([existing_df, pd.DataFrame(dqn_results)], ignore_index=True)
    combined_df.to_csv("results/dqn_benchmark_results.csv", index=False)
    print("[Report Saved] results/dqn_benchmark_results.csv")

    # ---- 7. Summary Table ----
    summary = combined_df.groupby('Method')[
        ['Legit Acceptance (%)', 'Abuse Blocked (%)', 'Avg Latency (ms)',
         'Avg CPU (%)', 'Overload Events', 'False Throttling (%)', 'Cumulative Reward']
    ].mean().round(2).reset_index()

    print("\n" + "=" * 100)
    print("  FULL BENCHMARK SUMMARY: DQN vs Tabular Q-Learning vs Industry Baselines")
    print("=" * 100)
    print(tabulate(summary, headers='keys', tablefmt='github', showindex=False))

    # ---- 8. Head-to-head Plot ----
    plot_head_to_head(combined_df)

    print("\n[DQN Training Complete] All artifacts saved:")
    print("  - models/dqn_weights.pkl")
    print("  - results/dqn_benchmark_results.csv")
    print("  - results/dqn_vs_tabular_training.png")
    print("  - results/dqn_benchmark_comparison.png")
