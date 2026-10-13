"""Recompute Fig. 4, Tables S5-S7 and the spatial statistics, then compare with the saved results."""

from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    for script in ['figure4.py', 'spatial.py', 'verify.py']:
        subprocess.run([sys.executable, str(ROOT / 'code' / script)], check=True, cwd=ROOT)
