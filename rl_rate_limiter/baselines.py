"""
Traditional API Rate Limiting Baselines:
1. Token Bucket
2. Leaky Bucket
3. Fixed Window Counter
4. Sliding Window Log / Counter
"""

import collections
from typing import Dict, Deque

class BaseRateLimiter:
    def allow_request(self, client_id: str, timestamp: float, cost: int = 1) -> bool:
        raise NotImplementedError

    def reset(self):
        pass


class TokenBucketLimiter(BaseRateLimiter):
    """
    Token Bucket Algorithm:
    Tokens accumulate at refill_rate up to capacity.
    Allows bursts up to capacity, then throttles to refill_rate.
    """
    def __init__(self, capacity: float = 30.0, refill_rate: float = 20.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens: Dict[str, float] = collections.defaultdict(lambda: capacity)
        self.last_update: Dict[str, float] = collections.defaultdict(float)

    def allow_request(self, client_id: str, timestamp: float, cost: int = 1) -> bool:
        last = self.last_update[client_id]
        if last == 0.0:
            last = timestamp
            self.last_update[client_id] = timestamp
            self.tokens[client_id] = self.capacity

        # Refill tokens
        elapsed = max(0.0, timestamp - last)
        self.tokens[client_id] = min(self.capacity, self.tokens[client_id] + elapsed * self.refill_rate)
        self.last_update[client_id] = timestamp

        if self.tokens[client_id] >= cost:
            self.tokens[client_id] -= cost
            return True
        return False

    def reset(self):
        self.tokens.clear()
        self.last_update.clear()


class LeakyBucketLimiter(BaseRateLimiter):
    """
    Leaky Bucket Algorithm:
    Requests enter a queue (bucket) of fixed capacity and leak at constant leak_rate.
    Smooths bursts into steady-state outflow.
    """
    def __init__(self, capacity: float = 30.0, leak_rate: float = 20.0):
        self.capacity = capacity
        self.leak_rate = leak_rate
        self.water: Dict[str, float] = collections.defaultdict(float)
        self.last_leak: Dict[str, float] = collections.defaultdict(float)

    def allow_request(self, client_id: str, timestamp: float, cost: int = 1) -> bool:
        last = self.last_leak[client_id]
        if last == 0.0:
            last = timestamp
            self.last_leak[client_id] = timestamp

        # Leak water
        elapsed = max(0.0, timestamp - last)
        self.water[client_id] = max(0.0, self.water[client_id] - elapsed * self.leak_rate)
        self.last_leak[client_id] = timestamp

        if self.water[client_id] + cost <= self.capacity:
            self.water[client_id] += cost
            return True
        return False

    def reset(self):
        self.water.clear()
        self.last_leak.clear()


class FixedWindowLimiter(BaseRateLimiter):
    """
    Fixed Window Counter:
    Resets counter every window_size seconds. Allows max max_requests per window.
    Suffers from boundary burst issue.
    """
    def __init__(self, max_requests: int = 25, window_size: float = 1.0):
        self.max_requests = max_requests
        self.window_size = window_size
        self.windows: Dict[str, int] = collections.defaultdict(int)
        self.counts: Dict[str, int] = collections.defaultdict(int)

    def allow_request(self, client_id: str, timestamp: float, cost: int = 1) -> bool:
        current_window = int(timestamp / self.window_size)
        if self.windows[client_id] != current_window:
            self.windows[client_id] = current_window
            self.counts[client_id] = 0

        if self.counts[client_id] + cost <= self.max_requests:
            self.counts[client_id] += cost
            return True
        return False

    def reset(self):
        self.windows.clear()
        self.counts.clear()


class SlidingWindowLimiter(BaseRateLimiter):
    """
    Sliding Window Log:
    Tracks timestamp of each admitted request in a deque.
    Removes timestamps older than window_size.
    """
    def __init__(self, max_requests: int = 25, window_size: float = 1.0):
        self.max_requests = max_requests
        self.window_size = window_size
        self.logs: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)

    def allow_request(self, client_id: str, timestamp: float, cost: int = 1) -> bool:
        dq = self.logs[client_id]
        threshold = timestamp - self.window_size

        while dq and dq[0] <= threshold:
            dq.popleft()

        if len(dq) + cost <= self.max_requests:
            for _ in range(cost):
                dq.append(timestamp)
            return True
        return False

    def reset(self):
        self.logs.clear()
