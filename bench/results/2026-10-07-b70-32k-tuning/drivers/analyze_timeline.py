"""Stream the captured request; correlate GPU ids to CPU submission scopes."""
import bisect
import collections
import json
from pathlib import Path

root = Path(__file__).resolve().parent
d = root / "15-workload-timeline-retry-02"
window = json.loads((d / "request-window.json").read_text())
lo, hi = window["start_us"], window["end_us"]
scopes = collections.defaultdict(list)
phases = collections.defaultdict(list)
boundaries = {}
source_count = 0
with (d / "source-events-request.jsonl").open("w") as out:
    for line in (d / "source-events.jsonl").open():
        v = json.loads(line)
        start, end = v["ts"], v["ts"] + v.get("dur", 0)
        if end < lo or start > hi:
            continue
        out.write(line)
        source_count += 1
        if v["name"].startswith("gemm."):
            scopes[v["tid"]].append((start, end, v["name"], v["args"]))
        elif v["name"] == "phase":
            phases[v["tid"]].append((start, v["args"]))
        elif v["name"] in ("request.prefill", "request.decode"):
            boundaries[v["name"]] = {"start_us": start, "end_us": end}
for array in (*scopes.values(), *phases.values()):
    array.sort(key=lambda x: x[0])
scope_starts = {tid: [x[0] for x in rows] for tid, rows in scopes.items()}
phase_starts = {tid: [x[0] for x in rows] for tid, rows in phases.items()}
print(json.dumps({"stage": "source_indexed", "request_source_events": source_count,
                  "gemm_scopes": sum(map(len, scopes.values())), "boundaries": boundaries}), flush=True)

def owner(tid, ts):
    if tid in scopes:
        index = bisect.bisect_right(scope_starts[tid], ts) - 1
        if index >= 0:
            v = scopes[tid][index]
            if v[0] <= ts <= v[1]:
                return {"scope": v[2], **v[3], "scope_start_us": v[0], "scope_end_us": v[1]}
    if tid in phases:
        index = bisect.bisect_right(phase_starts[tid], ts) - 1
        if index >= 0:
            return {"scope": None, **phases[tid][index][1]}
    return None

flows = {}
gpu = []
api = collections.defaultdict(lambda: [0, 0.0])
counts = collections.Counter()
trace = next(d.glob("strata.*.json"))
with (d / "request-timeline.json").open("w") as out:
    out.write('{"traceEvents":[\n')
    first = True
    for line in trace.open():
        if not line.startswith("{"):
            continue
        try:
            v = json.loads(line.rstrip().rstrip(","))
        except json.JSONDecodeError:
            continue
        start = v.get("ts", -1)
        end = start + v.get("dur", 0)
        if v.get("ph") != "M" and (end < lo or start > hi):
            continue
        counts[(v.get("cat", ""), v.get("ph", ""))] += 1
        if not first:
            out.write(",\n")
        out.write(json.dumps(v, separators=(",", ":")))
        first = False
        if v.get("ph") == "s" and v.get("cat", "").startswith("Flow_H2D_"):
            flows[str(v["id"])] = (v["tid"], start)
        elif v.get("cat") == "gpu_op" and v.get("ph") == "X":
            name = v["name"]
            kind = "copy" if "AppendMemoryCopy" in name else "fill" if "AppendMemoryFill" in name else "compute"
            gpu.append((start, end, kind, name, str(v.get("args", {}).get("id")), v["pid"], v["tid"]))
        elif v.get("cat") == "cpu_op" and v.get("ph") == "X":
            api[v["name"]][0] += 1
            api[v["name"]][1] += v.get("dur", 0)
    # Source annotations use exactly the same raw clock.
    for line in (d / "source-events-request.jsonl").open():
        if not first:
            out.write(",\n")
        out.write(line.strip())
        first = False
    out.write('\n]}\n')
print(json.dumps({"stage": "timeline_indexed", "gpu_events": len(gpu), "flows": len(flows)}), flush=True)

def union(intervals):
    result = []
    for start, end in sorted(intervals):
        start, end = max(start, lo), min(end, hi)
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result

def duration(intervals):
    return sum(e - s for s, e in intervals) / 1000

compute = union((g[0], g[1]) for g in gpu if g[2] == "compute")
copy = union((g[0], g[1]) for g in gpu if g[2] == "copy")
all_active = union((g[0], g[1]) for g in gpu)
both_union = union(compute + copy)
overlap_ms = duration(compute) + duration(copy) - duration(both_union)
shape_groups = collections.defaultdict(lambda: [0, 0.0])
kernel_groups = collections.defaultdict(lambda: [0, 0.0])
gemm_total = gemm_matched = 0
gemm_total_us = gemm_matched_us = 0.0
with (d / "gpu-correlation.jsonl").open("w") as out:
    for start, end, kind, name, kid, pid, tid in gpu:
        relation = flows.get(kid)
        source = owner(*relation) if relation else None
        kernel_groups[name][0] += 1
        kernel_groups[name][1] += end - start
        is_gemm = name.startswith("gemm_")
        if is_gemm:
            gemm_total += 1
            gemm_total_us += end - start
            if source and source.get("scope"):
                gemm_matched += 1
                gemm_matched_us += end - start
                key = tuple(source.get(k) for k in ("scope", "phase", "T", "N", "K", "ldx", "ldy", "quant_type"))
                shape_groups[key][0] += 1
                shape_groups[key][1] += end - start
        out.write(json.dumps({"start_us": start, "end_us": end, "kind": kind, "kernel": name,
                              "id": kid, "pid": pid, "tid": tid, "source": source}) + "\n")

result = {
    "class": "resident_workload_profile; instrumented intervals only",
    "window": window, "phase_boundaries": boundaries,
    "event_counts": {str(k): v for k, v in counts.items()},
    "gpu_events": len(gpu), "gpu_active_union_ms": duration(all_active),
    "compute_union_ms": duration(compute), "copy_union_ms": duration(copy),
    "compute_copy_overlap_ms": overlap_ms,
    "gpu_idle_within_instrumented_window_ms": (hi-lo)/1000-duration(all_active),
    "gemm_correlation": {"events": gemm_total, "exact_host_scope_events": gemm_matched,
                         "device_ms": gemm_total_us/1000, "matched_device_ms": gemm_matched_us/1000,
                         "coverage_pct": 100*gemm_matched/gemm_total if gemm_total else None},
    "top_kernels": [{"kernel": k, "calls": v[0], "device_ms": v[1]/1000}
                    for k,v in sorted(kernel_groups.items(), key=lambda x:-x[1][1])[:20]],
    "gemm_shapes": [{"scope": k[0], "phase": k[1], "T": k[2], "N": k[3], "K": k[4],
                      "ldx": k[5], "ldy": k[6], "quant_type": k[7], "calls": v[0], "device_ms": v[1]/1000}
                    for k,v in sorted(shape_groups.items(), key=lambda x:-x[1][1])],
    "host_api": [{"name": k, "calls": v[0], "cpu_ms": v[1]/1000}
                 for k,v in sorted(api.items(), key=lambda x:-x[1][1])[:20]],
    "limits": ["Profiling changes queue feeding and gaps substantially; union/overlap values are not unprofiled utilization or speedup.",
               "Opaque replay/cross-thread events without a matching host scope retain missing attribution; no guessed dimensions.",
               "Mapped-host reads inside GPU kernels are not explicit memcpy events; this trace does not measure total PCIe traffic.",
               "Hardware utilization/bandwidth counters remain unavailable; no roofline claim.",
               "Host phase predecessor labels without an enclosing GEMM scope are contextual only."]
}
(d / "timeline-summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ("gpu_events", "gpu_active_union_ms", "compute_union_ms", "copy_union_ms", "compute_copy_overlap_ms", "gemm_correlation")}, indent=2), flush=True)
