"""
Backend Server Simulator.
Models realistic microservice/backend system health dynamics:
- CPU utilization (cpu_t)
- Memory utilization (mem_t)
- Request queue length (q_t)
- Response latency (l_t) via M/M/1 queuing theory
- Error rate (err_t) under severe stress/overload
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class BackendMetrics:
    cpu_percent: float
    memory_percent: float
    latency_ms: float
    error_rate: float
    queue_length: int
    is_overloaded: bool

class BackendSimulator:
    def __init__(
        self,
        service_capacity_rps: float = 300.0,
        baseline_latency_ms: float = 25.0,
        baseline_cpu: float = 20.0,
        baseline_mem: float = 35.0,
        seed: int = 42
    ):
        self.capacity_rps = service_capacity_rps
        self.baseline_latency = baseline_latency_ms
        self.baseline_cpu = baseline_cpu
        self.baseline_mem = baseline_mem
        self.rng = np.random.default_rng(seed)

        self.current_queue = 0
        self.current_cpu = baseline_cpu
        self.current_mem = baseline_mem
        self.consecutive_overloads = 0

    def reset(self):
        self.current_queue = 0
        self.current_cpu = self.baseline_cpu
        self.current_mem = self.baseline_mem
        self.consecutive_overloads = 0

    def step(self, admitted_requests_count: int, dt: float = 1.0) -> BackendMetrics:
        """
        Simulates backend state transition over duration dt given admitted request volume.
        """
        arrival_rps = admitted_requests_count / max(dt, 0.001)
        utilization = arrival_rps / self.capacity_rps

        # 1. CPU Dynamics: smooth exponential approach towards load-dependent CPU
        target_cpu = self.baseline_cpu + (utilization * 65.0)
        if utilization > 1.0:
            target_cpu += (utilization - 1.0) * 80.0
        target_cpu = float(np.clip(target_cpu + self.rng.normal(0, 2.0), 5.0, 100.0))
        # Autoregressive smoothing (inertia of server metrics)
        self.current_cpu = 0.65 * self.current_cpu + 0.35 * target_cpu

        # 2. Queue Length Dynamics
        processed = int(self.capacity_rps * dt)
        net_arrivals = admitted_requests_count - processed
        self.current_queue = max(0, self.current_queue + net_arrivals)

        # 3. Memory Dynamics (grows with queue and active threads)
        target_mem = self.baseline_mem + (self.current_queue * 0.05) + (self.current_cpu * 0.25)
        target_mem = float(np.clip(target_mem, 10.0, 98.0))
        self.current_mem = 0.8 * self.current_mem + 0.2 * target_mem

        # 4. Latency Dynamics (M/M/1 Queueing Curve: W = 1 / (mu - lambda))
        if utilization < 0.95:
            # Normal queueing delay
            queue_delay_factor = 1.0 / max(0.05, (1.0 - utilization))
            latency = self.baseline_latency * queue_delay_factor
        else:
            # Severe congestion delay + queue backlog
            latency = self.baseline_latency * 20.0 + (self.current_queue * 1.5)
        # Add jitter
        latency = float(np.clip(latency + self.rng.normal(0, 4.0), self.baseline_latency, 2500.0))

        # 5. Error Rate (timeouts / HTTP 503 drops when CPU > 85% or latency > 500ms)
        is_overloaded = bool(self.current_cpu >= 85.0 or latency >= 600.0)
        if is_overloaded:
            self.consecutive_overloads += 1
            error_rate = min(0.95, 0.15 + (self.current_cpu - 85.0) * 0.04 + (self.consecutive_overloads * 0.05))
        else:
            self.consecutive_overloads = max(0, self.consecutive_overloads - 1)
            error_rate = max(0.001, self.rng.uniform(0.001, 0.02))

        return BackendMetrics(
            cpu_percent=round(self.current_cpu, 2),
            memory_percent=round(self.current_mem, 2),
            latency_ms=round(latency, 2),
            error_rate=round(error_rate, 4),
            queue_length=self.current_queue,
            is_overloaded=is_overloaded
        )
