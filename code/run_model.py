"""Run the minimal model (Figs. S9-S10) in outputs/minimal_model."""

from pathlib import Path
import shutil, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    out = ROOT / 'outputs/minimal_model'
    out.mkdir(parents=True, exist_ok=True)
    for name in ['run_model.py', 'parameters.json']:
        shutil.copy2(ROOT / 'models/minimal' / name, out / name)
    subprocess.run([sys.executable, str(out / 'run_model.py')], check=True)
