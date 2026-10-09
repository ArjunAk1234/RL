"""
Traffic Generator and Client Risk Scoring Engine.
Simulates realistic API client traffic streams across multiple behavioral personas:
- Normal legitimate users
- Flash-crowd / Bursty legitimate users
- Malicious / Abusive clients (DDoS, scraping, credential stuffing)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ClientRequest:
    client_id: str
    timestamp: float
    is_malicious: bool
    payload_size_kb: float
    endpoint: str
    risk_score: float

class ClientProfile:
    def __init__(self, client_id: str, client_type: str):
        self.client_id = client_id
        self.client_type = client_type  # 'legitimate_normal', 'legitimate_burst', 'abusive_bot'
        self.total_requests: int = 0
        self.recent_rate: float = 0.0
        self.calculated_risk: float = 0.05 if 'legitimate' in client_type else 0.85

    def reset(self):
        self.total_requests = 0
        self.recent_rate = 0.0
        self.calculated_risk = 0.05 if 'legitimate' in self.client_type else 0.85

    def update_risk(self, arrival_rate: float) -> float:
        """
        Calculates dynamic client risk score c_t in [0.0, 1.0] based on:
        - Arrival rate smoothing
        - Persona profile baseline
        """
        self.recent_rate = 0.7 * self.recent_rate + 0.3 * arrival_rate
        
        if self.client_type == 'abusive_bot':
            base_risk = 0.72 + min(0.26, self.recent_rate * 0.002)
        elif self.client_type == 'legitimate_burst':
            base_risk = 0.18 + min(0.25, self.recent_rate * 0.001)
        else:
            base_risk = 0.05 + min(0.12, self.recent_rate * 0.0005)
            
        self.calculated_risk = float(np.clip(base_risk, 0.0, 1.0))
        return self.calculated_risk


class TrafficGenerator:
    """
    Generates synthetic API request streams representing varied real-world scenarios.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.clients: Dict[str, ClientProfile] = {}
        self._init_client_pool()

    def reset(self):
        for profile in self.clients.values():
            profile.reset()

    def _init_client_pool(self):
        # 8 Normal legit clients
        for i in range(8):
            cid = f"legit_norm_{i}"
            self.clients[cid] = ClientProfile(cid, 'legitimate_normal')
        # 4 Bursty legit clients
        for i in range(4):
            cid = f"legit_burst_{i}"
            self.clients[cid] = ClientProfile(cid, 'legitimate_burst')
        # 4 Abusive bot clients
        for i in range(4):
            cid = f"abusive_bot_{i}"
            self.clients[cid] = ClientProfile(cid, 'abusive_bot')

    def generate_step_traffic(
        self,
        current_time: float,
        dt: float,
        scenario: str = 'mixed'
    ) -> List[ClientRequest]:
        """
        Generates requests arriving in time window [current_time, current_time + dt].
        Scenarios: 'normal', 'burst', 'abuse', 'overload', 'mixed'
        """
        requests: List[ClientRequest] = []

        for cid, profile in self.clients.items():
            ctype = profile.client_type
            
            # Determine expected arrival rate lambda (req/sec) per client
            if scenario == 'normal':
                rate = self.rng.uniform(6, 14) if ctype == 'legitimate_normal' else 0.5
            elif scenario == 'burst':
                if ctype == 'legitimate_burst':
                    rate = self.rng.uniform(60, 110)  # Sudden legitimate surge
                elif ctype == 'legitimate_normal':
                    rate = self.rng.uniform(8, 20)
                else:
                    rate = 1.0
            elif scenario == 'abuse':
                if ctype == 'abusive_bot':
                    rate = self.rng.uniform(70, 130)  # Aggressive volumetric attack
                else:
                    rate = self.rng.uniform(5, 12)
            elif scenario == 'overload':
                rate = self.rng.uniform(35, 75)
            elif scenario == 'mixed':
                if ctype == 'legitimate_normal':
                    rate = self.rng.uniform(8, 18)
                elif ctype == 'legitimate_burst':
                    rate = self.rng.uniform(40, 90) if (int(current_time) % 20 < 8) else self.rng.uniform(5, 12)
                else:
                    rate = self.rng.uniform(50, 100) if (int(current_time) % 16 < 9) else self.rng.uniform(10, 20)
            else:
                rate = 10.0

            # Poisson arrival count for this time window
            n_reqs = int(self.rng.poisson(rate * dt))
            profile.total_requests += n_reqs
            risk = profile.update_risk(rate)
            is_mal = (ctype == 'abusive_bot')

            # Generate requests vectorially
            if n_reqs > 0:
                timestamps = current_time + np.sort(self.rng.uniform(0, dt, size=n_reqs))
                for t in timestamps:
                    requests.append(ClientRequest(
                        client_id=cid,
                        timestamp=float(t),
                        is_malicious=is_mal,
                        payload_size_kb=float(self.rng.uniform(1.0, 15.0)),
                        endpoint="/api/v1/resource",
                        risk_score=risk
                    ))

        # Sort chronologically
        requests.sort(key=lambda r: r.timestamp)
        return requests
