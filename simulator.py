"""
Live RL Rate Limiting Simulator Backend
========================================
FastAPI server that:
 Run with:  uvicorn simulator:app --host 0.0.0.0 --port 8080 --reload
 1. Runs a real-time traffic simulation with 16 virtual clients (3 personas)
 2. Applies the trained Q-Learning policy every 1-second epoch
 3. Streams all events to the frontend via Server-Sent Events (SSE)
 4. Exposes REST endpoints for control (start/stop/scenario/reset)
"""

import asyncio
import json
import math
import pickle
import time
import os
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ───────────────────────────────────────────────────────────────────────────────
# Q-Table Loading
# ───────────────────────────────────────────────────────────────────────────────

QTABLE_PATH = "models/q_table.pkl"
ACTION_NAMES = {
    0: "Increase (+25%)",
    1: "Maintain",
    2: "Decrease (-25%)",
    3: "Throttle (Risk-Aware)",
    4: "Block High-Risk"
}
ACTION_COLORS = {0: "#3fb950", 1: "#58a6ff", 2: "#d29922", 3: "#f0883e", 4: "#f85149"}


def load_qtable():
    if os.path.exists(QTABLE_PATH):
        with open(QTABLE_PATH, "rb") as f:
            data = pickle.load(f)
        return data.get("q_table", {})
    return {}


Q_TABLE = load_qtable()


def discretize(r, b, c, cpu, lat) -> tuple:
    r_bin   = 0 if r   < 60   else (1 if r   < 180 else 2)
    b_bin   = 0 if b   < 1.3  else 1
    c_bin   = 0 if c   < 0.35 else (1 if c   < 0.65 else 2)
    cpu_bin = 0 if cpu < 65.0 else (1 if cpu < 85.0 else 2)
    lat_bin = 0 if lat < 50.0 else (1 if lat < 150.0 else 2)
    return (r_bin, b_bin, c_bin, cpu_bin, lat_bin)


def get_action_and_qvals(state_idx: tuple):
    if state_idx in Q_TABLE:
        qv = Q_TABLE[state_idx].tolist()
        action = int(np.argmax(qv))
    else:
        qv = [0.0] * 5
        action = 1  # Maintain
    return action, qv


# ───────────────────────────────────────────────────────────────────────────────
# Virtual Client Model
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class VirtualClient:
    client_id: str
    persona: str          # "normal" | "bursty" | "bot"
    base_rate: float      # requests/sec
    risk_score: float
    total_sent: int       = 0
    admitted:   int       = 0
    blocked:    int       = 0
    last_status: str      = "idle"   # "admitted" | "blocked" | "idle"
    last_action_taken: str = ""
    recent_rates: List[float] = field(default_factory=list)

    @property
    def acceptance_pct(self):
        return round(self.admitted / max(1, self.total_sent) * 100, 1)

    @property
    def block_pct(self):
        return round(self.blocked / max(1, self.total_sent) * 100, 1)


# ───────────────────────────────────────────────────────────────────────────────
# Simulation Engine
# ───────────────────────────────────────────────────────────────────────────────

class SimulationEngine:
    EPOCH_SEC = 1.0

    def __init__(self):
        self.rng = np.random.default_rng(42)
        self.running = False
        self.scenario = "mixed"
        self.clients: Dict[str, VirtualClient] = {}
        self._build_clients()

        # Aggregate metrics
        self.epoch_admitted = 0
        self.epoch_blocked  = 0
        self.total_admitted = 0
        self.total_blocked  = 0
        self.elapsed_sec    = 0

        # Backend simulation
        self.cpu      = 20.0
        self.latency  = 25.0
        self.queue    = 0
        self.err_rate = 0.01

        # RL policy state
        self.rate_mult  = 1.0
        self.base_quota = 25
        self.action     = 1
        self.qvals      = [0.0] * 5
        self.state_idx  = (0, 0, 0, 0, 0)

        # Time-series history (last 60 points)
        self.ts_admitted: deque = deque(maxlen=60)
        self.ts_blocked:  deque = deque(maxlen=60)
        self.ts_cpu:      deque = deque(maxlen=60)
        self.ts_latency:  deque = deque(maxlen=60)
        self.ts_labels:   deque = deque(maxlen=60)

        # Activity log
        self.activity_log: deque = deque(maxlen=40)

        # SSE subscribers
        self._subscribers: List[asyncio.Queue] = []

        self.rate_history: deque = deque(maxlen=10)

    def _build_clients(self):
        self.clients = {}
        personas = [
            ("normal",  8,  (6, 14),   0.05),
            ("bursty",  4,  (40, 90),  0.18),
            ("bot",     4,  (70, 130), 0.72),
        ]
        for persona, count, rate_range, base_risk in personas:
            for i in range(count):
                cid = f"{persona}_{i+1}"
                rate = float(self.rng.uniform(*rate_range))
                self.clients[cid] = VirtualClient(
                    client_id=cid,
                    persona=persona,
                    base_rate=rate,
                    risk_score=base_risk
                )

    def _update_client_risk(self, client: VirtualClient, current_rate: float):
        """Exponential moving average risk update."""
        if client.persona == "bot":
            base = 0.72 + min(0.26, current_rate * 0.002)
        elif client.persona == "bursty":
            base = 0.18 + min(0.25, current_rate * 0.001)
        else:
            base = 0.05 + min(0.12, current_rate * 0.0005)
        client.risk_score = float(np.clip(base, 0.0, 1.0))

    def _scenario_rate(self, client: VirtualClient) -> float:
        p = client.persona
        t = self.elapsed_sec
        sc = self.scenario

        if sc == "normal":
            if p == "normal": return float(self.rng.uniform(6, 14))
            if p == "bursty": return float(self.rng.uniform(2, 6))
            return float(self.rng.uniform(1, 3))

        elif sc == "burst":
            if p == "bursty": return float(self.rng.uniform(80, 140))
            if p == "normal": return float(self.rng.uniform(10, 25))
            return float(self.rng.uniform(2, 5))

        elif sc == "abuse":
            if p == "bot":    return float(self.rng.uniform(80, 150))
            if p == "normal": return float(self.rng.uniform(5, 12))
            return float(self.rng.uniform(5, 15))

        elif sc == "overload":
            return float(self.rng.uniform(40, 90))

        else:  # mixed
            if p == "normal": return float(self.rng.uniform(8, 18))
            if p == "bursty": return float(self.rng.uniform(50, 100) if t % 20 < 8 else self.rng.uniform(5, 12))
            return float(self.rng.uniform(55, 110) if t % 16 < 9 else self.rng.uniform(10, 22))

    def _simulate_backend(self, admitted_rps: float):
        capacity = 300.0
        util = admitted_rps / capacity

        # CPU
        target_cpu = 20.0 + util * 65.0 + (max(0, util - 1.0) * 80.0)
        target_cpu = float(np.clip(target_cpu + self.rng.normal(0, 1.5), 5, 100))
        self.cpu = 0.7 * self.cpu + 0.3 * target_cpu

        # Queue
        net = int(admitted_rps - capacity)
        self.queue = max(0, self.queue + net)

        # Latency (M/M/1)
        if util < 0.95:
            self.latency = 25.0 / max(0.05, 1.0 - util) + float(self.rng.uniform(-3, 3))
        else:
            self.latency = min(2500, 500.0 + self.queue * 1.5)
        self.latency = max(25.0, self.latency)

        # Error rate
        if self.cpu >= 85.0:
            self.err_rate = min(0.95, 0.1 + (self.cpu - 85.0) * 0.04)
        else:
            self.err_rate = max(0.001, 0.01)

    def _apply_rl_action(self):
        """Run one RL control-plane epoch."""
        # Aggregate state
        total_this_epoch = self.epoch_admitted + self.epoch_blocked
        r_t = total_this_epoch / self.EPOCH_SEC
        self.rate_history.append(r_t)
        avg_r = float(np.mean(self.rate_history)) if self.rate_history else 1.0
        b_t = r_t / max(avg_r, 1.0)
        c_t = float(np.mean([c.risk_score for c in self.clients.values()]))

        state_idx = discretize(r_t, b_t, c_t, self.cpu, self.latency)
        action, qvals = get_action_and_qvals(state_idx)

        # Update multiplier
        if action == 0:   self.rate_mult = min(2.5, self.rate_mult + 0.25)
        elif action == 2: self.rate_mult = max(0.3, self.rate_mult - 0.25)
        elif action == 3: self.rate_mult = max(0.2, self.rate_mult * 0.7)
        elif action == 4: self.rate_mult = max(0.1, self.rate_mult * 0.5)

        self.action    = action
        self.qvals     = qvals
        self.state_idx = state_idx

        self._log(f"RL Epoch | State {state_idx} → Action: {ACTION_NAMES[action]} | "
                  f"Multiplier: {self.rate_mult:.2f}x | CPU: {self.cpu:.1f}% | Lat: {self.latency:.0f}ms",
                  level="agent")

    def _admit_client_requests(self, client: VirtualClient, n_reqs: int) -> tuple[int, int]:
        effective_quota = max(1, int(self.base_quota * self.rate_mult))
        action = self.action
        risk   = client.risk_score

        if action == 4 and risk >= 0.35:
            return 0, n_reqs

        if action == 3:
            adj = max(1, int(effective_quota * (1.0 - 0.8 * risk)))
            admitted = min(n_reqs, adj)
        else:
            quota = effective_quota
            if action == 2 and risk >= 0.65:
                quota = max(1, quota // 2)
            admitted = min(n_reqs, quota)

        return admitted, n_reqs - admitted

    def _log(self, msg: str, level: str = "info"):
        entry = {"ts": round(self.elapsed_sec, 1), "msg": msg, "level": level}
        self.activity_log.append(entry)

    async def _run_epoch(self):
        """Execute one 1-second simulation epoch."""
        self.epoch_admitted = 0
        self.epoch_blocked  = 0

        for client in self.clients.values():
            rate = self._scenario_rate(client)
            self._update_client_risk(client, rate)
            n_reqs = int(self.rng.poisson(rate * self.EPOCH_SEC))
            client.total_sent += n_reqs

            adm, blk = self._admit_client_requests(client, n_reqs)
            client.admitted += adm
            client.blocked  += blk
            client.last_status = "admitted" if adm > 0 else ("blocked" if blk > 0 else "idle")
            client.last_action_taken = ACTION_NAMES[self.action]

            self.epoch_admitted += adm
            self.epoch_blocked  += blk

        self._simulate_backend(float(self.epoch_admitted))
        self._apply_rl_action()

        self.total_admitted += self.epoch_admitted
        self.total_blocked  += self.epoch_blocked
        self.elapsed_sec    += self.EPOCH_SEC

        # Time series
        t_label = int(self.elapsed_sec)
        self.ts_admitted.append(self.epoch_admitted)
        self.ts_blocked.append(self.epoch_blocked)
        self.ts_cpu.append(round(self.cpu, 1))
        self.ts_latency.append(round(self.latency, 1))
        self.ts_labels.append(t_label)

    async def run_loop(self):
        self.running = True
        while self.running:
            t0 = asyncio.get_event_loop().time()
            await self._run_epoch()
            snapshot = self._snapshot()
            for q in list(self._subscribers):
                try:
                    q.put_nowait(snapshot)
                except asyncio.QueueFull:
                    pass
            elapsed = asyncio.get_event_loop().time() - t0
            await asyncio.sleep(max(0.0, self.EPOCH_SEC - elapsed))

    def _snapshot(self) -> dict:
        return {
            "t":              self.elapsed_sec,
            "scenario":       self.scenario,
            "action":         self.action,
            "action_name":    ACTION_NAMES[self.action],
            "action_color":   ACTION_COLORS[self.action],
            "qvals":          self.qvals,
            "state_idx":      list(self.state_idx),
            "rate_mult":      round(self.rate_mult, 3),
            "eff_quota":      int(self.base_quota * self.rate_mult),
            "cpu":            round(self.cpu, 1),
            "latency":        round(self.latency, 1),
            "queue":          self.queue,
            "err_rate":       round(self.err_rate * 100, 2),
            "epoch_admitted": self.epoch_admitted,
            "epoch_blocked":  self.epoch_blocked,
            "total_admitted": self.total_admitted,
            "total_blocked":  self.total_blocked,
            "accept_pct":     round(self.total_admitted / max(1, self.total_admitted + self.total_blocked) * 100, 1),
            "clients": [
                {
                    "id":          c.client_id,
                    "persona":     c.persona,
                    "risk":        round(c.risk_score, 3),
                    "rate":        round(self._scenario_rate(c), 1),
                    "admitted":    c.admitted,
                    "blocked":     c.blocked,
                    "total":       c.total_sent,
                    "accept_pct":  c.acceptance_pct,
                    "status":      c.last_status,
                    "action":      c.last_action_taken,
                }
                for c in self.clients.values()
            ],
            "ts": {
                "labels":   list(self.ts_labels),
                "admitted": list(self.ts_admitted),
                "blocked":  list(self.ts_blocked),
                "cpu":      list(self.ts_cpu),
                "latency":  list(self.ts_latency),
            },
            "log": list(self.activity_log)[-15:]
        }

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=5)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self._subscribers:
            self._subscribers.remove(q)

    def stop(self):
        self.running = False

    def reset(self):
        self.stop()
        self.__init__()


# ───────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ───────────────────────────────────────────────────────────────────────────────

engine = SimulationEngine()
app = FastAPI(title="RL Rate Limiter Simulator", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_sim_task: Optional[asyncio.Task] = None


@app.post("/api/start")
async def start(scenario: str = "mixed"):
    global _sim_task
    engine.scenario = scenario
    if not engine.running:
        _sim_task = asyncio.create_task(engine.run_loop())
    return {"status": "started", "scenario": scenario}


@app.post("/api/stop")
async def stop():
    engine.stop()
    return {"status": "stopped"}


@app.post("/api/reset")
async def reset():
    global _sim_task
    engine.reset()
    return {"status": "reset"}


@app.post("/api/scenario/{sc}")
async def change_scenario(sc: str):
    engine.scenario = sc
    engine._log(f"Scenario changed → {sc.upper()}", level="system")
    return {"scenario": sc}


@app.get("/api/state")
async def state():
    return JSONResponse(engine._snapshot())


@app.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    q = engine.subscribe()

    async def stream() -> AsyncGenerator[str, None]:
        try:
            # Send initial state immediately
            snap = engine._snapshot()
            yield f"data: {json.dumps(snap)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    snap = await asyncio.wait_for(q.get(), timeout=2.0)
                    yield f"data: {json.dumps(snap)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            engine.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Serve static files (frontend)
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("simulator:app", host="0.0.0.0", port=8080, reload=False)
