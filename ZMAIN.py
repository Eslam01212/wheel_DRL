#!/usr/bin/env python3
import subprocess, sys

PY = sys.executable

# run all stages
for c in [
    [PY, "d.py"],
    [PY, "d2.py"],

]:
    subprocess.run(c, check=True)

print("\n✅ PIPELINE DONE")
