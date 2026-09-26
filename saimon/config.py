"""Repository layout: every data path in the package resolves through here.

ROOT is the repository root; DATA is the shipped derived-data tree
(data/); OUT is the working/archive tree (outputs/).  Override with the
SAIMON_ROOT environment variable to run the package against another copy
of the data (e.g. a full raw-data tree, see fetch/README.md).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SAIMON_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OPTICS = DATA / "optics"
DIGITIZED = DATA / "digitized"
MODELS = DATA / "models"
GLOSSAC_NC = DATA / "glossac" / "GloSSAC_V2.23_subset.nc"
TRACE_GASES = DATA / "trace_gases"
ACE = DATA / "ace"
ACE_FLOOR = DATA / "ace_floor"
CACHE = ROOT / "cache"
