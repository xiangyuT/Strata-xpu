"""Observation-only IPC timestamp tap around the unchanged Strata frontend."""
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/src/Strata")
from serve import server

path = Path(os.environ["STRATA_IPC_TIMESTAMPS"])
lock = threading.Lock()
active = None
sequence = 0


class Tap:
    def __init__(self, stream, input_side=False):
        self.stream = stream
        self.input_side = input_side

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def write(self, text):
        global active, sequence
        if text.startswith(("GEN ", "GENI ")):
            now = time.perf_counter_ns()
            epoch = time.time_ns()
            raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            with lock:
                sequence += 1
                active = {"sequence": sequence, "send_perf_ns": now, "send_epoch_ns": epoch, "send_raw_ns": raw,
                          "reader_pid": os.getpid(), "events": []}
        return self.stream.write(text)

    def readline(self, *args):
        global active
        line = self.stream.readline(*args)
        now = time.perf_counter_ns()
        raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        if line.startswith(("T ", "PP ", "REUSED ", "RESUME ", "DONE ")):
            with lock:
                if active is not None:
                    active["events"].append({"perf_ns": now, "raw_ns": raw, "line": line.rstrip("\n"),
                                              "reader_tid": threading.get_native_id()})
                    if line.startswith("DONE "):
                        active["done_perf_ns"] = now
                        with path.open("a") as out:
                            out.write(json.dumps(active) + "\n")
                        active = None
        return line


original_popen = server.popen


def tapped_popen(what, args, **kwargs):
    proc = original_popen(what, args, **kwargs)
    if what == "the Strata engine":
        proc.stdin = Tap(proc.stdin, True)
        proc.stdout = Tap(proc.stdout)
    return proc


server.popen = tapped_popen
sys.exit(server.main())
