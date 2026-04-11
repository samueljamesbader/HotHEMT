import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

def test_cli():
    scripts_dir = sysconfig.get_path('scripts')
    env = {**os.environ, 'PATH': scripts_dir + os.pathsep + os.environ.get('PATH', '')}

    out_dir='output/hemt3d_example'
    if Path(out_dir).exists():
        shutil.rmtree(out_dir)
    assert subprocess.run([sys.executable, str(Path(__file__).parent.parent/'examples/example.py'), '--no_plots'], env=env).returncode==0
    assert Path(out_dir).exists()

    assert subprocess.run('hhplot print 0 --sim_work_dir output/hemt3d_example', shell=True, env=env).returncode==0

if __name__ == "__main__":
    test_cli()