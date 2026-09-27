"""round53_tracegas_driver.py -- re-run the trace-gas budget scripts with the exact
edge-grid onion-peel gain (saimon.onion_peel, referee RC1 M1, round 53).

Same mechanics as round43_tracegas_driver.py: scripts run in-process, hapi.fetch
is blocked so nothing can stall on hitran.org, logs go to
outputs/round53_tracegas/<script>.log.  With --full-cache the untrimmed HITRAN
cache is used (needed by the far-infrared scripts, which read HNO3); otherwise
the round-43 trimmed cache (O3, N2O, H2O, CH4, CO2, CO) is used.

Run from the repo root:
  python reproduce/round53_tracegas_driver.py [--full-cache] "script.py [args]" ...
"""
import contextlib, io, os, runpy, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data/trace_gases/hitran_cache"
TRIM = ROOT / "outputs/round43_tracegas/hitran_cache_trim"
LOGS = ROOT / "outputs/round53_tracegas"
LOGS.mkdir(parents=True, exist_ok=True)
args = sys.argv[1:]
full = "--full-cache" in args
args = [a for a in args if a != "--full-cache"]
use = CACHE if full else TRIM
os.environ["SAGE_HITRAN_CACHE"] = str(use)
import saimon.trace_gases as tg
tg.HITRAN_CACHE = str(use)
import hapi
def _blocked(*a, **k):
    raise RuntimeError(f"hitran.org fetch blocked by round53 driver: {a}")
hapi.fetch = _blocked
os.chdir(ROOT)
for item in args:
    parts = item.split()
    script, sargs = parts[0], parts[1:]
    log = LOGS / (Path(script).stem + ("_" + "_".join(a.strip("-") for a in sargs) if sargs else "") + ".log")
    t0 = time.time()
    print(f"=== {script} {' '.join(sargs)} -> {log.name}", flush=True)
    buf = io.StringIO()
    sys.argv = [str(ROOT / "scripts" / script)] + sargs
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")
        status = "ok"
    except SystemExit as e:
        status = f"exit {e.code}"
    except Exception:
        buf.write(traceback.format_exc()); status = "FAILED"
    log.write_text(buf.getvalue())
    print(f"    {status} in {time.time() - t0:.0f} s", flush=True)
print("DONE", flush=True)
