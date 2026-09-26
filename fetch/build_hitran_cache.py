"""Build the HITRAN-derived cross-section cache from hitran.org (full chain).

The package computes line-by-line Voigt cross-section grids on demand
(saimon.trace_gases) from HITRAN2020 line lists fetched through HAPI, and
caches them under data/trace_gases/xsec_cache/ (the line lists themselves
go to data/trace_gases/hitran_cache/).  The repository ships the grids the
paper's steps use; everything else is rebuilt on first use once
`hitran-api` is installed:

    pip install hitran-api
    python fetch/build_hitran_cache.py            # runs the HITRAN-dependent steps once

Building takes hours for the heavy gases (O3, H2O, CH4 over wide windows)
and needs a working connection to hitran.org; fetches that stall can leave
an empty table behind -- delete data/trace_gases/hitran_cache/<gas>*.data
and rerun if a gas comes out with zero absorption.  Gordon et al. (2022,
JQSRT 277, 107949) is the reference for HITRAN2020.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["floorw", "snr", "limb", "cont", "cfc", "highres", "oe", "stT", "lines", "fig1"]

if __name__ == "__main__":
    try:
        import hapi  # noqa: F401
    except ImportError:
        sys.exit("hitran-api is not installed: pip install hitran-api")
    sys.exit(subprocess.run([sys.executable, str(ROOT / "make_all.py"), "--only", *STEPS], cwd=ROOT).returncode)
