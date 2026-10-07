"""Derive the aligned report from completed per-request records."""
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
source = Path(sys.argv[2])
public_dir = source / "bench/results/2026-09-30-community-rtx-5090"
public_rows = {row["label"]: row for row in json.loads((public_dir / "results.json").read_text())}
report = {"plan": json.loads((root / "experiment-plan-v2.json").read_text()),
          "outcomes": json.loads((root / "run-outcomes.json").read_text()),
          "arms": {}, "limitations": [
              "Development evidence, not formal acceptance or a GPU-only comparison.",
              "CUDA community reports use different source versions, host hardware and engine policies.",
              "Our server fixes the otherwise omitted request seed to 42.",
              "Each length has an excluded warmup beyond the public harness's short warmup.",
              "128000-token requests, vision and general answer quality were not measured.",
              "SYCL hit_rate and pcie_share counters are unavailable; zero ram_blobs does not prove zero PCIe mirror traffic.",
              "Primary GPU-count arm order was 2gpu, 4gpu, then the matching fixed-pcie 1gpu baseline.",
              "Borrow-setting records were collected earlier; they are exploratory and use a different container identity from the final baseline.",
              "Earlier automatic-probe no-borrow prefill differs from the matching explicit-parameter baseline; small gains should not be treated as stable speedups."
          ]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for file in ("run-outcomes-followup.json", "run-outcomes-multi.json", "run-outcomes-baseline.json"):
    if (root / file).exists():
        report["outcomes"].update(json.loads((root / file).read_text()))
if (root / "experiment-plan-multi.json").exists():
    report["multi_gpu_plan"] = json.loads((root / "experiment-plan-multi.json").read_text())
if (root / "experiment-plan-multi-v2.json").exists():
    report["multi_gpu_plan"] = json.loads((root / "experiment-plan-multi-v2.json").read_text())
report["primary_gpu_count_baseline"] = "1gpu"
for arm, outcome in report["outcomes"].items():
    directory = root / arm
    config = json.loads((directory / "strata-iq2_xs.json").read_text())
    if outcome["status"] != "passed":
        report["arms"][arm] = {"status": "failed", "outcome": outcome,
                              "configuration": json.loads((directory / "strata-iq2_xs.json").read_text())}
        continue
    rows = json.loads((directory / "results.json").read_text())
    measured = [r for r in rows if r["label"] != "warmup" and not r["label"].endswith("-warmup")]
    startup = (directory / "engine.log").read_text()
    slots = re.search(r"expert cache (\d+) slots, ([\d.]+) GiB", startup)
    mirror = re.search(r"(\d+) of (\d+) experts missing from VRAM mirrored.*?\(([\d.]+) GiB", startup)
    probe = re.search(r"PCIe probe: ([\d.]+) GB/s.*?pcie_frac ([\d.]+)", startup)
    chunk = re.search(r"prompt path[^\n]*", startup)
    comparisons = []
    for row in measured:
        reference = public_rows[row["label"]]
        captured_body = (directory / (row["label"] + "-request.json")).read_bytes().rstrip(b"\n")
        if hashlib.sha256(captured_body).hexdigest() != row["request_sha256"]:
            raise RuntimeError(f"captured body digest mismatch: {arm} {row['label']}")
        comparisons.append({"label": row["label"], "sha256": row["request_sha256"],
                            "matches_5090_request_bytes": row["request_sha256"] == reference["request_sha256"]})
        if not comparisons[-1]["matches_5090_request_bytes"]:
            raise RuntimeError(f"request body mismatch: {arm} {row['label']}")
    summary = json.loads((directory / "summary.json").read_text())
    for length, values in summary.items():
        subset = [row for row in measured if row["engine"]["prompt_tokens"] == int(length)]
        accepted = [100 * r["engine"]["drafts_accepted"] / r["engine"]["drafts_offered"] for r in subset]
        values["draft_acceptance_pct"] = {"median": statistics.median(accepted),
                                        "min": min(accepted), "max": max(accepted)}
    report["arms"][arm] = {
        "configuration": config,
        "configuration_sha256": digest(directory / "strata-iq2_xs.json"),
        "summary": summary, "measured": measured, "excluded_warmups": len(rows) - len(measured),
        "request_identity": comparisons,
        "smoke": json.loads((directory / "smoke-results.json").read_text()) if (directory / "smoke-results.json").exists() else None,
        "startup": {"expert_slots": int(slots[1]) if slots else None,
                    "expert_cache_GiB": float(slots[2]) if slots else None,
                    "mirrored_experts": int(mirror[1]) if mirror else None,
                    "mirror_GiB": float(mirror[3]) if mirror else None,
                    "probe_GB_s": float(probe[1]) if probe else None,
                    "effective_pcie_frac": float(probe[2]) if probe else float(config["args"][config["args"].index("--pcie-frac") + 1]) if "--pcie-frac" in config["args"] else None,
                    "prompt_path": chunk[0] if chunk else None,
                    "stage_lines": [line for line in startup.splitlines() if "layer split:" in line]}
    }
report["setting_comparison"] = {}
for length in ("4096", "32768"):
    a = report["arms"]["1gpu"]["summary"][length]
    b = report["arms"]["borrow-fixed-pcie"]["summary"][length]
    report["setting_comparison"][length] = {
        key + "_median_change_pct": 100 * (b[key]["median"] / a[key]["median"] - 1)
        for key in ("prefill_tok_s", "decode_tok_s", "client_ttft_s", "client_elapsed_s")
    }
    paired = []
    arows = {r["label"]: r for r in report["arms"]["1gpu"]["measured"]}
    for brow in report["arms"]["borrow-fixed-pcie"]["measured"]:
        if brow["engine"]["prompt_tokens"] != int(length):
            continue
        arow = arows[brow["label"]]
        paired.append({"label": brow["label"],
                       "prefill_change_pct": 100 * (arow["engine"]["prompt_ms"] / brow["engine"]["prompt_ms"] - 1),
                       "decode_change_pct": 100 * (arow["engine"]["decode_ms"] / brow["engine"]["decode_ms"] - 1)})
    report["setting_comparison"][length]["per_request"] = paired
report["gpu_scaling"] = {}
for arm in ("2gpu", "4gpu"):
    if arm not in report["arms"] or report["outcomes"][arm]["status"] != "passed":
        continue
    report["gpu_scaling"][arm] = {}
    for length in ("4096", "32768"):
        baseline = report["arms"]["1gpu"]["summary"][length]
        candidate = report["arms"][arm]["summary"][length]
        report["gpu_scaling"][arm][length] = {
            "prefill_ratio": candidate["prefill_tok_s"]["median"] / baseline["prefill_tok_s"]["median"],
            "decode_ratio": candidate["decode_tok_s"]["median"] / baseline["decode_tok_s"]["median"],
            "elapsed_ratio": baseline["client_elapsed_s"]["median"] / candidate["client_elapsed_s"]["median"]}
report["community_reference"] = {
    "rtx5090": {"source": "https://github.com/Niko1221/Strata/blob/main/bench/results/2026-09-30-community-rtx-5090/README.md",
                "source_revision": "d6708a4aae15b4860000d54c8af9e84d684bce09", "context": 131072,
                "summary": {key: value for key, value in json.loads((public_dir / "summary.json").read_text()).items()
                            if key in ("4096", "32768")}},
    "rtx5070ti": {"source": "https://github.com/obiscr/Strata/blob/36ddda197895b09895b5bcbb956813b35fce219b/bench/results/2026-10-06-community-rtx5070ti-5900x/README.md",
                  "engine_version": "0.1.39", "context": 65536,
                  "summary": {"4096": {"prefill_tok_s_median": 1992, "decode_tok_s_median": 102.0},
                              "32768": {"prefill_tok_s_median": 2817, "decode_tok_s_median": 101.3}}}
}
telemetry_path = root / "telemetry.jsonl"
if telemetry_path.exists():
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    gpu_memory = [metric["value"] for row in telemetry for metric in row.get("gpu", {}).get("device_level", [])
                  if metric["metrics_type"] == "XPUM_STATS_MEMORY_USED"]
    report["sampled_telemetry"] = {
        "samples": len(telemetry), "nominal_sleep_s": 5,
        "gpu_memory_MiB_max": max(gpu_memory) if gpu_memory else None,
        "system_used_GiB_max": max((r["memory"]["MemTotal_KiB"] - r["memory"]["MemAvailable_KiB"]) / 1048576 for r in telemetry),
        "swap_used_MiB_min": min((r["memory"]["SwapTotal_KiB"] - r["memory"]["SwapFree_KiB"]) / 1024 for r in telemetry),
        "swap_used_MiB_max": max((r["memory"]["SwapTotal_KiB"] - r["memory"]["SwapFree_KiB"]) / 1024 for r in telemetry),
        "note": "System-wide sampled values include other services; not process RSS or continuous peaks."
    }
if (root / "telemetry-multi.csv").exists():
    with (root / "telemetry-multi.csv").open() as stream:
        samples = list(csv.DictReader(stream, skipinitialspace=True))
    report["multi_gpu_sampled_memory"] = {
        str(device): max(float(r["GPU Memory Used (MiB)"]) for r in samples if int(r["DeviceId"]) == device)
        for device in range(4)}
    report["multi_gpu_sampled_memory_note"] = "MiB, system-wide sampled maximum across both multi-card arms; not a continuous peak."
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"setting_comparison": report["setting_comparison"],
                  "startup": {k: v["startup"] for k, v in report["arms"].items()},
                  "telemetry": report.get("sampled_telemetry")}, indent=2))
