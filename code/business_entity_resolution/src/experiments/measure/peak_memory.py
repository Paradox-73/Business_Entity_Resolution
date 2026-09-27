"""Peak resident memory of every running Python process, sampled once per second (used for
src/logs/build_final_memory.txt while build_final.py ran).

  python peak_memory.py <seconds> <out.txt>        (start it, then start the run to measure in another shell)

Writes <out.txt> after every sample: one line per process command line (last 80 characters), largest peak first.
Needs psutil.
"""
import sys
import time

import psutil

peak = {}
end = time.time() + float(sys.argv[1])
out = sys.argv[2]
while time.time() < end:
    for p in psutil.process_iter(["name", "memory_info", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if p.info["name"] and p.info["name"].lower().startswith("python") and "peak_memory" not in cmd:
                k = " ".join((p.info["cmdline"] or [])[1:3])[-80:]
                peak[k] = max(peak.get(k, 0), p.info["memory_info"].rss / 2 ** 30)
        except Exception:
            pass
    with open(out, "w") as f:
        for k, v in sorted(peak.items(), key=lambda x: -x[1]):
            f.write(f"{v:6.2f} GB  {k}\n")
    time.sleep(1)
