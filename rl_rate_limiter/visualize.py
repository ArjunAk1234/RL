"""
Visualization and Plotting Suite.
Generates publication-ready figures for research papers and project presentations:
1. RL Training Reward & Exploration Decay
2. Dynamic Response Timeline (Burst/Attack Stress Test)
3. Comparative Benchmarks across all algorithms
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Configure styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_training_curve(rewards: List[float], epsilons: List[float], save_path: str = "results/training_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Rolling average of reward
    window = max(5, len(rewards) // 20)
    rolling_reward = pd.Series(rewards).rolling(window=window, min_periods=1).mean()

    ax1.plot(rewards, color='#6baed6', alpha=0.35, label='Episode Reward')
    ax1.plot(rolling_reward, color='#1f77b4', linewidth=2.0, label=f'Moving Avg (w={window})')
    ax1.set_ylabel('Cumulative Reward', fontsize=11, fontweight='bold')
    ax1.set_title('RL Agent Training Convergence & Exploration Decay', fontsize=13, fontweight='bold', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right')

    ax2.plot(epsilons, color='#d95f02', linewidth=1.8, label=r'Exploration Rate ($\epsilon$)')
    ax2.set_xlabel('Episode Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Epsilon ($\epsilon$)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Training curve: {save_path}")


def plot_dynamic_timeline(
    timeline_rl: Dict[str, List[float]],
    timeline_baseline: Dict[str, List[float]],
    save_path: str = "results/dynamic_stress_timeline.png"
):
    """
    Plots side-by-side dynamic response of RL vs Token Bucket under a sudden traffic burst/attack.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    steps = np.arange(len(timeline_rl["cpu"]))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # 1. Traffic Admitted vs Blocked
    ax1.plot(steps, timeline_rl["admitted_legit"], label='Legit Admitted (RL)', color='#2ca02c', linewidth=2)
    ax1.plot(steps, timeline_rl["blocked_abuse"], label='Abuse Blocked (RL)', color='#d62728', linestyle='--', linewidth=2)
    ax1.plot(steps, timeline_baseline["admitted_legit"], label='Legit Admitted (Token Bucket)', color='#1f77b4', alpha=0.6)
    ax1.set_ylabel('Requests / sec', fontsize=10, fontweight='bold')
    ax1.set_title('Dynamic Stress Test: Adaptive RL vs Static Token Bucket', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 2. CPU Utilization
    ax2.plot(steps, timeline_rl["cpu"], label='Backend CPU (RL)', color='#2ca02c', linewidth=2.2)
    ax2.plot(steps, timeline_baseline["cpu"], label='Backend CPU (Token Bucket)', color='#d62728', linewidth=1.8, linestyle='--')
    ax2.axhline(85, color='darkred', linestyle=':', label='Overload Threshold (85%)', alpha=0.8)
    ax2.set_ylabel('CPU Utilization (%)', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 3. Latency (ms)
    ax3.plot(steps, timeline_rl["latency"], label='Latency (RL)', color='#2ca02c', linewidth=2.2)
    ax3.plot(steps, timeline_baseline["latency"], label='Latency (Token Bucket)', color='#d62728', linewidth=1.8, linestyle='--')
    ax3.set_xlabel('Time (Seconds)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Latency (ms)', fontsize=10, fontweight='bold')
    ax3.legend(loc='upper right', frameon=True)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Dynamic stress timeline: {save_path}")


def plot_benchmark_bars(df: pd.DataFrame, save_path: str = "results/benchmark_comparison.png"):
    """
    Produces publication bar charts comparing key metrics across all methods.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Average across all scenarios per method
    summary = df.groupby('Method')[['Legit Acceptance (%)', 'Abuse Blocked (%)', 'Avg Latency (ms)', 'Overload Events']].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    palette = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

    metrics = [
        ('Legit Acceptance (%)', 'Legitimate Request Acceptance (%)', axes[0, 0]),
        ('Abuse Blocked (%)', 'Abuse / Bot Blocking Rate (%)', axes[0, 1]),
        ('Avg Latency (ms)', 'Average Response Latency (ms)', axes[1, 0]),
        ('Overload Events', 'Total Backend Overload Incidents', axes[1, 1])
    ]

    for col, title, ax in metrics:
        bars = ax.bar(summary['Method'], summary[col], color=palette[:len(summary)], edgecolor='black', alpha=0.85, width=0.55)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xticks(range(len(summary['Method'])))
        ax.set_xticklabels(summary['Method'], rotation=18, ha='right', fontsize=9)
        # Add value tags on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle('Comprehensive Performance Evaluation: RL vs Industry Baselines', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Benchmark comparison bars: {save_path}")
