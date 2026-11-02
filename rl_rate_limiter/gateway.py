"""
FastAPI Live Proxy Middleware — Risk-Aware Adaptive Rate Limiter
=================================================================
Wraps the trained Q-Learning policy in a fully asynchronous HTTP reverse proxy.

Architecture:
  [Client] → [FastAPI Gateway: /proxy/*] → [Admission Controller (RL Policy)] → [Upstream Target]
                                                  ↕
                                          [Backend Metrics Monitor]

Endpoints:
  POST /proxy/*      — Proxy any HTTP request through the RL rate limiter
  GET  /metrics      — Real-time gateway metrics (JSON)
  GET  /policy       — Current rate-limiting policy state
  GET  /dashboard    — Live HTML dashboard (auto-refreshes every 2 seconds)
  POST /admin/reset  — Reset all gateway counters

Run:
  source .venv/bin/activate
  uvicorn rl_rate_limiter.gateway:app --host 0.0.0.0 --port 8000 --reload

Test with curl:
  curl -X POST http://localhost:8000/proxy/resource -H "X-Client-ID: user_1"
  curl http://localhost:8000/metrics
  curl http://localhost:8000/dashboard
"""

import time
import asyncio
import pickle
import os
from collections import defaultdict, deque
from typing import Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.background import BackgroundTask
import uvicorn

# ---------------------------------------------------------------------------
# Load Trained Policy (with graceful fallback)
# ---------------------------------------------------------------------------

QTABLE_PATH = "models/q_table.pkl"

def load_policy():
    if os.path.exists(QTABLE_PATH):
        with open(QTABLE_PATH, "rb") as f:
            data = pickle.load(f)
        return data.get("q_table", {})
    return {}

Q_TABLE: Dict = load_policy()

def q_table_action(state_idx: tuple) -> int:
    """Look up optimal action from trained Q-table. Falls back to Maintain."""
    if not Q_TABLE or state_idx not in Q_TABLE:
        return 1  # Fallback: Maintain
    q_vals = Q_TABLE[state_idx]
    return int(np.argmax(q_vals))

def discretize(obs: dict) -> tuple:
    """Replicates env.py discretize_state() for inference."""
    r   = obs["r_t"]
    b   = obs["b_t"]
    c   = obs["c_t"]
    cpu = obs["cpu_t"]
    lat = obs["l_t"]

    r_bin   = 0 if r < 60 else (1 if r < 180 else 2)
    b_bin   = 0 if b < 1.3 else 1
    c_bin   = 0 if c < 0.35 else (1 if c < 0.65 else 2)
    cpu_bin = 0 if cpu < 65.0 else (1 if cpu < 85.0 else 2)
    lat_bin = 0 if lat < 50.0 else (1 if lat < 150.0 else 2)
    return (r_bin, b_bin, c_bin, cpu_bin, lat_bin)

# ---------------------------------------------------------------------------
# Gateway State — Thread-Safe In-Memory Telemetry
# ---------------------------------------------------------------------------

WINDOW_SEC = 1.0  # Control-plane decision epoch

class GatewayState:
    def __init__(self):
        # Per-client request history (sliding window)
        self.client_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.client_risk:       Dict[str, float]  = defaultdict(lambda: 0.1)

        # Aggregate gateway counters
        self.epoch_admissions: int = 0
        self.epoch_rejections: int = 0
        self.total_admitted:   int = 0
        self.total_rejected:   int = 0
        self.total_requests:   int = 0

        # Simulated backend telemetry (updated per control epoch)
        self.cpu_sim:     float = 20.0
        self.latency_sim: float = 25.0
        self.queue_sim:   int   = 0
        self.error_rate:  float = 0.0

        # Rate limit policy (controlled by RL agent)
        self.rate_limit_multiplier: float = 1.0
        self.base_quota_per_client: int   = 25

        # History for burstiness calculation
        self.rate_history: deque = deque(maxlen=10)

        # Recent activity log for dashboard
        self.recent_log: deque = deque(maxlen=50)

        # Control plane epoch tracking
        self.last_epoch_ts: float = time.monotonic()
        self.last_action:   int   = 1  # Maintain
        self.last_state:    tuple = (0, 0, 0, 0, 0)

        self.lock = asyncio.Lock()

    def compute_risk(self, client_id: str, current_time: float) -> float:
        history = self.client_timestamps[client_id]
        window_reqs = sum(1 for t in history if current_time - t <= 5.0)
        freq = window_reqs / 5.0

        is_bot = client_id.startswith("bot_") or client_id.startswith("attacker")
        if is_bot:
            risk = min(0.98, 0.72 + freq * 0.002)
        else:
            risk = min(0.45, 0.05 + freq * 0.003)

        self.client_risk[client_id] = risk
        return risk

    def compute_obs(self, current_time: float) -> dict:
        elapsed = max(current_time - self.last_epoch_ts, 0.001)
        r_t = self.epoch_admissions / elapsed
        self.rate_history.append(r_t)
        avg_r = float(np.mean(self.rate_history)) if self.rate_history else 1.0
        b_t = r_t / max(avg_r, 1.0)
        c_t = float(np.mean(list(self.client_risk.values()))) if self.client_risk else 0.1

        return {
            "r_t":   r_t,
            "b_t":   b_t,
            "c_t":   c_t,
            "cpu_t": self.cpu_sim,
            "mem_t": min(95.0, 30.0 + self.cpu_sim * 0.5),
            "l_t":   self.latency_sim,
            "err_t": self.error_rate,
            "q_t":   float(self.queue_sim),
            "limit_mult": self.rate_limit_multiplier,
        }

    def simulate_backend(self):
        """Simulates M/M/1 backend degradation based on admitted request volume."""
        capacity = 300.0
        rps = self.epoch_admissions / max(self.last_epoch_ts - (time.monotonic() - WINDOW_SEC), 0.01)
        utilization = rps / capacity

        # CPU dynamics
        target_cpu = 20.0 + utilization * 65.0
        if utilization > 1.0:
            target_cpu += (utilization - 1.0) * 80.0
        self.cpu_sim = 0.7 * self.cpu_sim + 0.3 * min(target_cpu, 100.0)

        # Queue dynamics
        net = self.epoch_admissions - int(capacity * WINDOW_SEC)
        self.queue_sim = max(0, self.queue_sim + net)

        # Latency via M/M/1
        if utilization < 0.95:
            self.latency_sim = 25.0 / max(0.05, 1.0 - utilization)
        else:
            self.latency_sim = min(2500.0, 500.0 + self.queue_sim * 1.5)

        # Error rate
        if self.cpu_sim >= 85.0:
            self.error_rate = min(0.95, 0.1 + (self.cpu_sim - 85.0) * 0.04)
        else:
            self.error_rate = 0.01

    async def run_control_plane(self):
        """Periodic RL policy update task (runs every WINDOW_SEC)."""
        while True:
            await asyncio.sleep(WINDOW_SEC)
            async with self.lock:
                current_time = time.monotonic()
                self.simulate_backend()

                obs = self.compute_obs(current_time)
                state_idx = discretize(obs)
                action = q_table_action(state_idx)

                # Apply action to policy multiplier
                if action == 0:
                    self.rate_limit_multiplier = min(2.5, self.rate_limit_multiplier + 0.25)
                elif action == 2:
                    self.rate_limit_multiplier = max(0.3, self.rate_limit_multiplier - 0.25)
                elif action == 3:
                    self.rate_limit_multiplier = max(0.2, self.rate_limit_multiplier * 0.7)
                elif action == 4:
                    self.rate_limit_multiplier = max(0.1, self.rate_limit_multiplier * 0.5)

                self.last_action  = action
                self.last_state   = state_idx
                self.last_epoch_ts = current_time

                # Reset epoch counters
                self.total_admitted += self.epoch_admissions
                self.total_rejected += self.epoch_rejections
                self.epoch_admissions = 0
                self.epoch_rejections = 0


state = GatewayState()
ACTION_NAMES = {
    0: "Increase (+25%)",
    1: "Maintain",
    2: "Decrease (-25%)",
    3: "Throttle (Risk-Aware)",
    4: "Block High-Risk"
}

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Risk-Aware Adaptive API Rate Limiter",
    description="RL-powered API gateway with real-time risk-aware admission control",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(state.run_control_plane())

# ---------------------------------------------------------------------------
# Proxy Endpoint
# ---------------------------------------------------------------------------

@app.api_route("/proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request):
    client_id = request.headers.get("X-Client-ID", request.client.host)
    current_time = time.monotonic()

    async with state.lock:
        state.client_timestamps[client_id].append(current_time)
        state.total_requests += 1

        risk = state.compute_risk(client_id, current_time)
        effective_quota = max(1, int(state.base_quota_per_client * state.rate_limit_multiplier))

        # Window count for this client
        window_count = sum(1 for t in state.client_timestamps[client_id]
                          if current_time - t <= WINDOW_SEC)

        # Admission decision
        action = state.last_action
        admitted = False

        if action == 4 and risk >= 0.35:
            admitted = False
        elif action == 3:
            adj_quota = max(1, int(effective_quota * (1.0 - 0.8 * risk)))
            admitted = window_count <= adj_quota
        else:
            if action == 2 and risk >= 0.65:
                effective_quota = max(1, effective_quota // 2)
            admitted = window_count <= effective_quota

        log_entry = {
            "ts":       round(current_time % 86400, 2),
            "client":   client_id,
            "risk":     round(risk, 3),
            "admitted": admitted,
            "action":   ACTION_NAMES.get(action, "?"),
            "path":     f"/{path}"
        }
        state.recent_log.append(log_entry)

        if admitted:
            state.epoch_admissions += 1
        else:
            state.epoch_rejections += 1

    if not admitted:
        return JSONResponse(
            status_code=429,
            content={
                "error":      "Too Many Requests",
                "client_id":  client_id,
                "risk_score": round(risk, 3),
                "policy":     ACTION_NAMES.get(action, "?"),
                "retry_after": f"{WINDOW_SEC:.1f}s"
            }
        )

    # Simulate upstream response (replace with httpx proxy in production)
    return JSONResponse(
        status_code=200,
        content={
            "status":    "admitted",
            "client_id": client_id,
            "risk_score": round(risk, 3),
            "path":      f"/{path}",
            "policy":    ACTION_NAMES.get(action, "?"),
            "multiplier": round(state.rate_limit_multiplier, 2)
        }
    )

# ---------------------------------------------------------------------------
# Metrics Endpoint
# ---------------------------------------------------------------------------

@app.get("/metrics", response_class=JSONResponse)
async def metrics():
    async with state.lock:
        obs = state.compute_obs(time.monotonic())
        total = state.total_admitted + state.total_rejected
        acceptance_rate = round(state.total_admitted / max(1, total) * 100, 2)
        return {
            "gateway": {
                "total_requests":   state.total_requests,
                "total_admitted":   state.total_admitted,
                "total_rejected":   state.total_rejected,
                "acceptance_rate":  f"{acceptance_rate}%",
            },
            "backend": {
                "cpu_utilization":  f"{obs['cpu_t']:.1f}%",
                "latency_ms":       f"{obs['l_t']:.1f}ms",
                "queue_depth":      state.queue_sim,
                "error_rate":       f"{obs['err_t'] * 100:.1f}%"
            },
            "rl_policy": {
                "current_action":   ACTION_NAMES.get(state.last_action, "?"),
                "rate_multiplier":  round(state.rate_limit_multiplier, 3),
                "effective_quota":  int(state.base_quota_per_client * state.rate_limit_multiplier),
                "state_bin":        state.last_state,
                "q_table_states":   len(Q_TABLE)
            },
            "risk": {
                "avg_client_risk":  round(float(np.mean(list(state.client_risk.values()))) if state.client_risk else 0.0, 3),
                "active_clients":   len(state.client_risk),
                "traffic_rps":      round(obs["r_t"], 1),
                "burstiness_index": round(obs["b_t"], 3)
            }
        }

# ---------------------------------------------------------------------------
# Policy Endpoint
# ---------------------------------------------------------------------------

@app.get("/policy", response_class=JSONResponse)
async def policy():
    async with state.lock:
        return {
            "action":         ACTION_NAMES.get(state.last_action, "?"),
            "action_id":      state.last_action,
            "multiplier":     round(state.rate_limit_multiplier, 3),
            "base_quota":     state.base_quota_per_client,
            "effective_quota": int(state.base_quota_per_client * state.rate_limit_multiplier),
            "state_idx":      state.last_state,
            "q_table_loaded": len(Q_TABLE) > 0,
            "q_table_states": len(Q_TABLE),
            "client_risks":   {k: round(v, 3) for k, v in list(state.client_risk.items())[-10:]}
        }

# ---------------------------------------------------------------------------
# Admin Reset
# ---------------------------------------------------------------------------

@app.post("/admin/reset")
async def reset_gateway():
    async with state.lock:
        state.epoch_admissions = 0
        state.epoch_rejections = 0
        state.total_admitted   = 0
        state.total_rejected   = 0
        state.total_requests   = 0
        state.client_risk.clear()
        state.client_timestamps.clear()
        state.rate_history.clear()
        state.recent_log.clear()
        state.cpu_sim     = 20.0
        state.latency_sim = 25.0
        state.queue_sim   = 0
        state.error_rate  = 0.0
        state.rate_limit_multiplier = 1.0
        state.last_action = 1
    return {"status": "reset", "message": "Gateway state cleared."}

# ---------------------------------------------------------------------------
# Live HTML Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>RL Rate Limiter — Live Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', monospace; padding: 20px; }
    h1  { color: #58a6ff; font-size: 1.5rem; margin-bottom: 6px; }
    p.sub { color: #8b949e; font-size: 0.85rem; margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; }
    .card h3 { color: #79c0ff; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }
    .val { font-size: 1.9rem; font-weight: 700; color: #e6edf3; }
    .val.green  { color: #3fb950; }
    .val.yellow { color: #d29922; }
    .val.red    { color: #f85149; }
    .val.blue   { color: #58a6ff; }
    .label { font-size: 0.75rem; color: #8b949e; margin-top: 4px; }
    .policy-box { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; margin-top: 14px; }
    .action-badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .log-table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 0.78rem; }
    .log-table th { background: #21262d; color: #8b949e; padding: 6px 10px; text-align: left; }
    .log-table td { padding: 5px 10px; border-top: 1px solid #21262d; }
    .admitted  { color: #3fb950; }
    .rejected  { color: #f85149; }
    .refresh { color: #8b949e; font-size: 0.73rem; margin-top: 16px; }
  </style>
</head>
<body>
  <h1>🛡️ Risk-Aware Adaptive API Rate Limiter</h1>
  <p class="sub">Live Dashboard — RL Policy Engine (Q-Learning) | Refreshes every 2s</p>

  <div class="grid" id="cards">
    <div class="card"><h3>Total Requests</h3><div class="val blue" id="total">—</div><div class="label">All-time requests</div></div>
    <div class="card"><h3>Acceptance Rate</h3><div class="val" id="accept">—</div><div class="label">Legitimate admitted %</div></div>
    <div class="card"><h3>Backend CPU</h3><div class="val" id="cpu">—</div><div class="label">Simulated server utilization</div></div>
    <div class="card"><h3>Latency</h3><div class="val" id="lat">—</div><div class="label">M/M/1 response time</div></div>
    <div class="card"><h3>Traffic (RPS)</h3><div class="val blue" id="rps">—</div><div class="label">Current request rate</div></div>
    <div class="card"><h3>Avg Client Risk</h3><div class="val" id="risk">—</div><div class="label">Mean risk score (0–1)</div></div>
    <div class="card"><h3>Queue Depth</h3><div class="val" id="queue">—</div><div class="label">Backend request backlog</div></div>
    <div class="card"><h3>Error Rate</h3><div class="val" id="err">—</div><div class="label">HTTP 5xx / 503 drops</div></div>
  </div>

  <div class="policy-box">
    <h3 style="color:#79c0ff;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">Current RL Policy</h3>
    <span class="action-badge" id="action-badge" style="background:#1f6feb22;border:1px solid #1f6feb;color:#79c0ff;">—</span>
    <span style="margin-left:16px;color:#8b949e;font-size:0.82rem;">Rate Multiplier: <strong id="mult" style="color:#e6edf3">—</strong></span>
    <span style="margin-left:16px;color:#8b949e;font-size:0.82rem;">Effective Quota: <strong id="quota" style="color:#e6edf3">—</strong> req/s</span>
    <span style="margin-left:16px;color:#8b949e;font-size:0.82rem;">State: <strong id="state-idx" style="color:#d2a8ff">—</strong></span>
  </div>

  <table class="log-table">
    <thead><tr>
      <th>Time</th><th>Client</th><th>Risk Score</th><th>Status</th><th>Policy Action</th><th>Path</th>
    </tr></thead>
    <tbody id="log-body"></tbody>
  </table>

  <p class="refresh" id="refresh-ts">Last updated: —</p>

<script>
  const colorCPU  = v => parseFloat(v) >= 85 ? 'red' : parseFloat(v) >= 65 ? 'yellow' : 'green';
  const colorRisk = v => parseFloat(v) >= 0.65 ? 'red' : parseFloat(v) >= 0.35 ? 'yellow' : 'green';
  const colorAccept = v => parseFloat(v) >= 80 ? 'green' : parseFloat(v) >= 60 ? 'yellow' : 'red';

  async function refresh() {
    try {
      const [m, p] = await Promise.all([
        fetch('/metrics').then(r => r.json()),
        fetch('/policy').then(r => r.json())
      ]);
      const g = m.gateway, b = m.backend, r = m.risk;
      const acc = parseFloat(g.acceptance_rate);

      document.getElementById('total').textContent   = g.total_requests.toLocaleString();
      document.getElementById('accept').textContent  = g.acceptance_rate;
      document.getElementById('accept').className    = 'val ' + colorAccept(acc);
      document.getElementById('cpu').textContent     = b.cpu_utilization;
      document.getElementById('cpu').className       = 'val ' + colorCPU(b.cpu_utilization);
      document.getElementById('lat').textContent     = b.latency_ms;
      document.getElementById('rps').textContent     = r.traffic_rps + ' rps';
      document.getElementById('risk').textContent    = r.avg_client_risk;
      document.getElementById('risk').className      = 'val ' + colorRisk(r.avg_client_risk);
      document.getElementById('queue').textContent   = b.queue_depth;
      document.getElementById('err').textContent     = b.error_rate;

      document.getElementById('action-badge').textContent = p.action;
      document.getElementById('mult').textContent   = p.multiplier;
      document.getElementById('quota').textContent  = p.effective_quota;
      document.getElementById('state-idx').textContent = JSON.stringify(p.state_idx);

      // Fetch log via policy (recent_log not exposed separately)
      document.getElementById('refresh-ts').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
    } catch(e) { console.error(e); }
  }

  refresh();
  setInterval(refresh, 2000);
</script>
</body>
</html>"""

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("rl_rate_limiter.gateway:app", host="0.0.0.0", port=8000, reload=True)
