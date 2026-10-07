"""Run a fresh-prompt Strata block inside the omix runtime."""
import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
p.add_argument("--samples", type=int, default=3)
p.add_argument("--warmups", type=int, default=1)
p.add_argument("--unitrace-session")
args = p.parse_args()
root = Path("/artifacts")
out = root / args.arm
cfg_path = out / "strata-iq2_xs.json"
cfg = json.loads(cfg_path.read_text())
url = "http://127.0.0.1:" + str(cfg["port"])
rows = []


def get(path):
    with urllib.request.urlopen(url + path, timeout=15) as r:
        return json.load(r)


def notify(event, **fields):
    item = {"epoch_s": time.time(), "arm": args.arm, "event": event, **fields}
    with (root / "notifications.jsonl").open("a") as stream:
        stream.write(json.dumps(item) + "\n")
    print(json.dumps(item), flush=True)


def perform(label, filename, measured):
    body = (root / "requests" / filename).read_bytes().rstrip(b"\n")
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    start_perf_ns = time.perf_counter_ns()
    start_epoch_ns = time.time_ns()
    start_raw_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    start = start_perf_ns / 1e9
    first = None
    last_content = None
    deltas = []
    chunks, text, reasoning = [], [], []
    finish = None
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            arrival_raw_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            arrival = time.perf_counter() - start
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            chunks.append(event)
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    first = first if first is not None else arrival
                    last_content = arrival
                    deltas.append({"arrival_s": arrival, "arrival_raw_ns": arrival_raw_ns, "text": delta["content"]})
                    text.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - start
    engine = get("/metrics")["requests"][0]
    if not text or reasoning or engine.get("reused", 0) != 0:
        raise RuntimeError("invalid output/reasoning/prompt reuse")
    if measured and (engine["prompt_tokens"] != 32768 or engine["engine_generated"] != 256 or finish != "length"):
        raise RuntimeError("wrong measured input/output boundary")
    ipc = json.loads((out / "ipc-timestamps.jsonl").read_text().splitlines()[-1])
    tokens = [e for e in ipc["events"] if e["line"].startswith("T ")]
    if measured and len(tokens) != engine["engine_generated"]:
        raise RuntimeError("IPC token count mismatch")
    n_tokens = len(tokens)
    latency = {"client_ttft_ms": first * 1000,
               "client_last_text_ms": last_content * 1000,
               "client_end_ms": elapsed * 1000,
               "client_post_text_tail_ms": (elapsed - last_content) * 1000,
               "client_tpot_ms": (last_content - first) * 1000 / (engine["engine_generated"] - 1) if engine["engine_generated"] > 1 else None,
               "client_delta_gaps_ms": [(b["arrival_s"] - a["arrival_s"]) * 1000 for a,b in zip(deltas, deltas[1:])],
               "client_content_events": len(deltas),
               "engine_average_ms_per_generated_token": engine["decode_ms"] / engine["engine_generated"],
               "engine_ipc_ttft_ms": (tokens[0]["perf_ns"] - ipc["send_perf_ns"]) / 1e6 if tokens else None,
               "engine_ipc_tpot_ms": (tokens[-1]["perf_ns"] - tokens[0]["perf_ns"]) / 1e6 / (n_tokens - 1) if n_tokens > 1 else None,
               "engine_ipc_token_gaps_ms": [(b["perf_ns"]-a["perf_ns"])/1e6 for a,b in zip(tokens,tokens[1:])],
               "definitions": "Client TPOT uses first-to-last nonempty text-delta arrival divided by engine-generated tokens minus one; chunking may coalesce initial tokens. IPC times are frontend reader arrival times, not producer compute times. Delta gaps are not token gaps."}
    row = {"client_clock_anchor": {"start_perf_ns": start_perf_ns, "start_epoch_ns": start_epoch_ns, "start_raw_ns": start_raw_ns}, "latency": latency, "label": label, "measured": measured, "engine": engine, "ttft_s": first,
           "elapsed_s": elapsed, "finish_reason": finish, "text": "".join(text)}
    (out / (label + "-raw.json")).write_text(json.dumps({"chunks": chunks, "deltas": deltas, "ipc": ipc, "row": row}, indent=2) + "\n")
    rows.append(row)
    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    notify("request_complete", label=label, prompt_ms=engine.get("prompt_ms"),
           decode_ms=engine.get("decode_ms"), elapsed_s=elapsed)


notify("server_start")
with (out / "server.log").open("w") as log:
    proc = subprocess.Popen(["/opt/strata-venv/bin/python", "-u", "/artifacts/latency_server.py",
                             "--engine", "strata", "--config", str(cfg_path),
                             "--host", "127.0.0.1", "--port", str(cfg["port"])],
                            cwd="/src/Strata", stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True,
                            env={**os.environ, "STRATA_IPC_TIMESTAMPS": str(out / "ipc-timestamps.jsonl")})
    (out / "server.pid").write_text(str(proc.pid) + "\n")
    try:
        deadline = time.monotonic() + 480
        while True:
            if proc.poll() is not None:
                raise RuntimeError("server exited before ready")
            try:
                health = get("/health")
                if health.get("loaded"):
                    (out / "health.json").write_text(json.dumps(health, indent=2) + "\n")
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("server readiness timeout")
            time.sleep(2)
        notify("server_ready")
        perform("short-warmup", "warmup.json", False)
        for n in range(args.warmups):
            perform(f"32k-warmup-{n+1}", "tokens-32768-warmup.json", False)
        if args.unitrace_session:
            subprocess.run(["/tools/unitrace/bin/unitrace", "--resume", args.unitrace_session], check=True)
        for n in range(1, args.samples + 1):
            perform(f"sample-{n}", f"tokens-32768-run-{n}.json", True)
        if args.unitrace_session:
            subprocess.run(["/tools/unitrace/bin/unitrace", "--pause", args.unitrace_session], check=True)
        (out / "status.json").write_text(json.dumps({"status": "passed", "samples": args.samples}, indent=2) + "\n")
        notify("block_complete", samples=args.samples)
    except Exception as error:
        (out / "status.json").write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2) + "\n")
        notify("block_failed", error=str(error))
        raise
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
