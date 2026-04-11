import os
import shutil
import sys
import sysconfig
from pathlib import Path
import subprocess

def test_cli():

    out_dir = 'output/hemt3d_example'
    if Path(out_dir).exists():
        shutil.rmtree(out_dir)

    # Run the example script
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "examples/example.py"), '--no_plots'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Example script failed with output:\n{result.stdout}\n{result.stderr}"
    assert Path(out_dir).exists()

    # Run the hhplot command
    hhplot = Path(sysconfig.get_path('scripts')) / 'hhplot'
    result = subprocess.run(
        [str(hhplot), 'print', '0', '--sim_work_dir', 'output/hemt3d_example'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"hhplot command failed with output:\n{result.stdout}\n{result.stderr}"

if __name__ == "__main__":
    test_cli()