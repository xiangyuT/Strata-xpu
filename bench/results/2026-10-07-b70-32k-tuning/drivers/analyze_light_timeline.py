import collections
import json
from pathlib import Path

root = Path(__file__).resolve().parent
d = root / "17-light-device-timeline"
raw = json.loads((d / "sample-1-raw.json").read_text())
ipc = raw["ipc"]
lo = ipc["send_raw_ns"] / 1000
hi = next(v["raw_ns"] / 1000 for v in ipc["events"] if v["line"].startswith("DONE "))
intervals = collections.defaultdict(list)
names = {}
engines = {}
trace = next(d.glob("strata.*.json"))
events = 0
for line in trace.open():
    if not line.startswith("{"):
        continue
    try:
        v = json.loads(line.rstrip().rstrip(","))
    except json.JSONDecodeError:
        continue
    if v.get("ph") == "M" and v.get("name") == "thread_name":
        engines[str(v["tid"])] = v["args"]["name"]
    if v.get("cat") != "gpu_op" or v.get("ph") != "X":
        continue
    start, end = v["ts"], v["ts"] + v["dur"]
    if end < lo or start > hi:
        continue
    name = v["name"]
    kind = "copy" if "AppendMemoryCopy" in name else "fill" if "AppendMemoryFill" in name else "compute"
    intervals[kind].append((max(start,lo), min(end,hi)))
    names.setdefault(kind, collections.Counter())[str(v["tid"])] += 1
    events += 1

def union(values):
    result = []
    for start, end in sorted(values):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result

def duration(values):
    return sum(e-s for s,e in values)/1000

compute = union(intervals["compute"])
copy = union(intervals["copy"])
all_active = union(sum(intervals.values(), []))
combined = union(compute+copy)
overlap = duration(compute)+duration(copy)-duration(combined)
heavy = json.loads((root / "15-workload-timeline-retry-02/timeline-summary-compact.json").read_text())
result = {"class":"resident_workload_profile; not performance data", "start_us":lo,"end_us":hi,
          "gpu_events":events,"gpu_active_union_ms":duration(all_active),
          "compute_union_ms":duration(compute),"copy_union_ms":duration(copy),
          "compute_copy_overlap_ms":overlap,
          "engine_metadata":engines,"kind_tracks":{k:dict(v) for k,v in names.items()},
          "heavy_trace_overlap_ms":heavy["compute_copy_overlap_ms"],
          "profiled_prompt_ms":raw["row"]["engine"]["prompt_ms"],
          "output_matches_unprofiled":raw["row"]["text"]==json.loads((root/"16-default-off-latency/sample-1-raw.json").read_text())["row"]["text"],
          "interpretation":"Both captures show copy activity on Compute Engine <0,0>. Near-zero measured overlap persists after removing full API and CPU-shape logging. The profiler still perturbs queue feeding; source/backend routing must be checked before making an unprofiled bottleneck or speedup claim.",
          "limits":"Explicit memcpy only; mapped-host PCIe reads are not counted. No hardware counters or unprofiled utilization estimate."}
(d/"timeline-summary.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2),flush=True)
