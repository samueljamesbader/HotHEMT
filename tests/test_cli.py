import os
import shutil
from pathlib import Path

def test_cli():

    out_dir='output/hemt3d_example'
    if Path(out_dir).exists():
        shutil.rmtree(out_dir)
    assert os.system(f'python "{str(Path(__file__).parent.parent/"examples/example.py")}" --no_plots')==0
    assert Path(out_dir).exists()
    
    assert os.system('hhplot print 0 --sim_work_dir output/hemt3d_example')==0

if __name__ == "__main__":
    test_cli()