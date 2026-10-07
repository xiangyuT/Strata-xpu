import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("before")
p.add_argument("candidate")
p.add_argument("after")
p.add_argument("--out", required=True)
args = p.parse_args()
root = Path(__file__).resolve().parent
arms = [args.before, args.candidate, args.after]
blocks, cfgs, slots = [], [], []
for arm in arms:
    d = root / arm
    assert json.loads((d / "status.json").read_text())["status"] == "passed"
    rows = [x for x in json.loads((d / "results.json").read_text()) if x["measured"]]
    assert [x["label"] for x in rows] == ["sample-1", "sample-2", "sample-3"]
    for x in rows:
        e = x["engine"]
        assert e["prompt_tokens"] == 32768 and e["engine_generated"] == 256
        assert e.get("reused", 0) == 0 and x["finish_reason"] == "length"
        assert x["latency"]["client_tpot_ms"] is not None
    blocks.append(rows)
    cfgs.append(json.loads((d / "strata-iq2_xs.json").read_text()))
    match = re.search(r"expert cache (\d+) slots", (d / "engine.log").read_text())
    assert match
    slots.append(int(match[1]))
assert len(set(slots)) == 1
assert cfgs[0]["args"] == cfgs[1]["args"] == cfgs[2]["args"]
assert cfgs[0]["sampling"] == cfgs[1]["sampling"] == cfgs[2]["sampling"]
assert cfgs[0]["env"] == cfgs[2]["env"]
env_delta = {k: v for k, v in cfgs[1]["env"].items() if cfgs[0]["env"].get(k) != v}

def value(row, key):
    if key in row["latency"]:
        return row["latency"][key]
    if key in row["engine"]:
        return row["engine"][key]
    return row[key]

metrics = {}
for key in ["client_ttft_ms", "client_tpot_ms", "engine_ipc_tpot_ms",
            "engine_average_ms_per_generated_token", "prompt_ms", "decode_ms", "elapsed_s"]:
    b, c, a = [[value(x, key) for x in rows] for rows in blocks]
    paired = [(x+y)/2 for x,y in zip(b,a)]
    changes = [100*(x/y-1) for x,y in zip(c,paired)]
    metrics[key] = {"baseline_before": b, "candidate": c, "baseline_after": a,
                    "paired_baseline": paired, "paired_change_pct": changes,
                    "paired_mean_change_pct": statistics.mean(changes),
                    "wins": sum(x<0 for x in changes),
                    "baseline_drift_pct": 100*(statistics.mean(a)/statistics.mean(b)-1),
                    "baseline_paired_mean": statistics.mean(paired),
                    "candidate_mean": statistics.mean(c)}
outputs = []
for i in range(3):
    texts = [b[i]["text"] for b in blocks]
    outputs.append({"sample": i+1, "equal": len(set(texts)) == 1,
                    "sha256": [hashlib.sha256(x.encode()).hexdigest() for x in texts]})
result = {"evidence_class": "development_performance_comparison", "arms": arms,
          "samples_per_block": 3, "actual_expert_cache_slots": slots[0],
          "config_env_delta": env_delta, "metrics": metrics, "outputs": outputs,
          "all_outputs_equal": all(x["equal"] for x in outputs),
          "tpot_definition": blocks[1][0]["latency"]["definitions"],
          "limits": "Development three-sample B/C/B, not formal acceptance; TTFT and first-to-last output TPOT reported separately."}
(root/args.out).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"output":args.out,"all_outputs_equal":result["all_outputs_equal"],"env_delta":env_delta,"metrics":metrics},indent=2))
