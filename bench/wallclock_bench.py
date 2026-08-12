"""Interleaved same-GPU A/B wall-clock bench.

Builds each mode's engine ONCE, then alternates A,B,A,B,... on the same GPU so
tenant/clock noise hits both sides symmetrically; report per-rep numbers and
let the caller pair them. Usage:
  wallclock_bench.py <modeA> <modeB> <temp> <maxbs> <nreq> <reps>
"""
import json
import sys

from treespark import engine as E

modeA, modeB = sys.argv[1], sys.argv[2]
temp, maxbs, nreq, reps = float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
seed0 = int(sys.argv[7]) if len(sys.argv) > 7 else 0
E.MAX_BS = maxbs
engines = {m: E.Engine(m, temperature=temp) for m in dict.fromkeys([modeA, modeB])}
out = {modeA: [], modeB: []}
for rep in range(reps):
    for m in (modeA, modeB):
        r = E.run(m, n_requests=nreq, seed=seed0 + rep, temperature=temp, engine=engines[m])
        out[m].append({"goodput": round(r["goodput"], 1), "draft": round(r["t_draft_s"], 2),
                       "verify": round(r["t_verify_s"], 2), "wall": round(r["wall_s"], 2)})
print("RESULT", json.dumps(out))
