"""Continuous-batching discrete-event simulation, driven by measured components.

Real pieces:   L(bs,k) measured step costs (results/serving_cost_model.json);
               per-round (tree_size, accepted) traces from real decoding (results/trace_*.json).
Simulated:     Poisson arrivals, admission up to MAX_BS, iteration-level scheduling
               (requests join/leave every step), per-step load-adaptive theta choice.
Excluded:      drafter/build cost (shape-independent per round; same convention as
               the static goodput table), prefill cost, kernel/runtime overheads.

Scheduler step: bs = |active|; each request consumes its next round (k_i, a_i);
step time = T_hat(bs, mean k_i). Requests complete at 150 generated tokens.

adaptive-load: theta picked per STEP from current bs (bs<=4 -> 0.02,
<=8 -> 0.05, else 0.15), matching the T=0 dynamic-engine controller; each
request's round is drawn from the SAME prompt's rounds under the chosen theta
config, preserving per-prompt difficulty correlation.
"""

import json
import os
import random

D = os.path.dirname(os.path.abspath(__file__))
MAX_BS = 64
TOKENS_PER_REQ = 150
SIM_MS, WARMUP_MS = 120_000.0, 20_000.0

_T = {int(bs): {int(k): v for k, v in d.items()}
      for bs, d in json.load(open(f"{D}/results/serving_cost_model.json")).items()}
_BSS = sorted(_T)
_KS = sorted(next(iter(_T.values())))


def _interp_k(row, k):
    if k <= _KS[0]:
        return row[_KS[0]]
    for a, b in zip(_KS, _KS[1:]):
        if a <= k <= b:
            w = (k - a) / (b - a)
            return row[a] * (1 - w) + row[b] * w
    a, b = _KS[-2], _KS[-1]                       # linear extrapolation beyond k=56
    return row[b] + (row[b] - row[a]) / (b - a) * (k - b)


def t_hat(bs, k):
    bs = min(bs, _BSS[-1])
    if bs <= _BSS[0]:
        return _interp_k(_T[_BSS[0]], k)
    for a, b in zip(_BSS, _BSS[1:]):
        if a <= bs <= b:
            w = (bs - a) / (b - a)
            return _interp_k(_T[a], k) * (1 - w) + _interp_k(_T[b], k) * w


def load_rounds(cfg):
    """-> list over prompts of list of (tree_size, accepted) rounds."""
    if cfg == "nospec":
        return [[(1, 1)] * TOKENS_PER_REQ]
    rows = json.load(open(f"{D}/results/trace_{cfg}.json"))["rows"]
    return [list(zip(r["tree_sizes"], r["accepted_lens"])) for r in rows]


THETAS = ["adapt0.02", "adapt0.05", "adapt0.15"]


def pick_theta(bs):
    return THETAS[0] if bs <= 4 else THETAS[1] if bs <= 8 else THETAS[2]


def simulate(cfg, lam, seed=0):
    """cfg: 'nospec' | 'chain7' | 'markov28' | 'markov56' | fixed 'adapt*' |
    'adaptive-load'. lam: arrivals per second."""
    rng = random.Random(seed)
    pools = {c: load_rounds(c) for c in THETAS} if cfg == "adaptive-load" else {cfg: load_rounds(cfg)}
    n_prompts = len(next(iter(pools.values())))

    now, next_arrival = 0.0, 0.0
    queue, active, done = [], [], []
    tokens_out, arrivals = 0, 0
    while now < SIM_MS:
        while next_arrival <= now:                # admit all arrivals up to now
            pid = rng.randrange(n_prompts)
            queue.append({"arrive": next_arrival, "pid": pid, "tok": 0, "round": 0})
            arrivals += 1
            next_arrival += rng.expovariate(lam) * 1000.0
        while queue and len(active) < MAX_BS:
            active.append(queue.pop(0))
        if not active:
            now = next_arrival
            continue

        bs = len(active)
        if cfg == "adaptive-load":
            rounds_of = pools[pick_theta(bs)]
        else:
            rounds_of = pools[cfg]
        ks = []
        for r in active:
            trace = rounds_of[r["pid"] % len(rounds_of)]
            if cfg == "adaptive-load":
                k, a = trace[rng.randrange(len(trace))]      # same-prompt random round
            else:
                k, a = trace[min(r["round"], len(trace) - 1)]
            r["_k"], r["_a"] = k, a
            ks.append(max(k, 1))
        now += t_hat(bs, sum(ks) / len(ks))

        still = []
        for r in active:
            r["tok"] += r["_a"]
            r["round"] += 1
            if r["tok"] >= TOKENS_PER_REQ:
                if now >= WARMUP_MS:
                    done.append(now - r["arrive"])
                    tokens_out += TOKENS_PER_REQ
            else:
                still.append(r)
        active = still

    span_s = (SIM_MS - WARMUP_MS) / 1000.0
    done.sort()
    return {
        "goodput": tokens_out / span_s,
        "completed": len(done),
        "p50_s": done[len(done) // 2] / 1000 if done else float("inf"),
        "p99_s": done[int(len(done) * 0.99)] / 1000 if done else float("inf"),
    }


if __name__ == "__main__":
    CFGS = [
        "nospec", "chain7", "markov28", "markov56", "adapt0.02",
        "adapt0.15", "adaptive-load",
    ]
    paper_results = {}
    for lam in (2, 5, 10, 15, 20, 25):
        print(f"\n=== lambda = {lam} req/s (offered load {lam * TOKENS_PER_REQ} tok/s) ===")
        print(f"{'config':<14} {'goodput':>8} {'done':>6} {'P50':>7} {'P99':>8}")
        paper_results[str(lam)] = {}
        for cfg in CFGS:
            r = simulate(cfg, lam)
            print(f"{cfg:<14} {r['goodput']:>8.0f} {r['completed']:>6} "
                  f"{r['p50_s']:>6.1f}s {r['p99_s']:>7.1f}s")
            if cfg in ("nospec", "chain7", "markov56", "adaptive-load"):
                paper_results[str(lam)][cfg] = r
    output_path = os.path.join(D, "results", "serving_sim_results.json")
    with open(output_path, "w") as f:
        json.dump(paper_results, f, indent=1)
        f.write("\n")
    print(f"\nwrote {output_path}")
