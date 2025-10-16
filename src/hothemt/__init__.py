__version__ = '0.0.1'

from contextlib import contextmanager
import sys
from pathlib import Path
import time
from typing import Optional

DEFAULT_SIM_WORK_DIR = Path(__file__).parent.parent.parent/"output"
def get_mesh_dir(SIM_WORK_DIR:Optional[Path|str]=None) -> Path:
    SIM_WORK_DIR = SIM_WORK_DIR or DEFAULT_SIM_WORK_DIR
    d = Path(SIM_WORK_DIR) / 'meshes'
    d.mkdir(parents=True, exist_ok=True)
    return d
def get_info_dir(SIM_WORK_DIR:Optional[Path|str]=None) -> Path:
    SIM_WORK_DIR = SIM_WORK_DIR or DEFAULT_SIM_WORK_DIR
    d = Path(SIM_WORK_DIR) / 'infos'
    d.mkdir(parents=True, exist_ok=True)
    return d
def get_state_dir(SIM_WORK_DIR:Optional[Path|str]=None) -> Path:
    SIM_WORK_DIR = SIM_WORK_DIR or DEFAULT_SIM_WORK_DIR
    d = Path(SIM_WORK_DIR) / 'states'
    d.mkdir(parents=True, exist_ok=True)
    return d
def get_log_dir(SIM_WORK_DIR:Optional[Path|str]=None) -> Path:
    SIM_WORK_DIR = SIM_WORK_DIR or DEFAULT_SIM_WORK_DIR
    d = Path(SIM_WORK_DIR) / 'logs'
    d.mkdir(parents=True, exist_ok=True)
    return d
def get_summary_path(summary_group: str, SIM_WORK_DIR:Optional[Path|str]=None) -> Path:
    SIM_WORK_DIR = SIM_WORK_DIR or DEFAULT_SIM_WORK_DIR
    return Path(SIM_WORK_DIR) / f'{summary_group}.csv'

@contextmanager
def specific_sim_work_dir(SIM_WORK_DIR:Path|str|None, clear: bool=False):
    global DEFAULT_SIM_WORK_DIR
    if SIM_WORK_DIR is None:
        assert clear is False, "Cannot clear default sim work dir"
        yield; return
    else:
        old=DEFAULT_SIM_WORK_DIR
        DEFAULT_SIM_WORK_DIR=Path(SIM_WORK_DIR)
        try:
            import shutil
            if clear and DEFAULT_SIM_WORK_DIR.exists():
                shutil.rmtree(DEFAULT_SIM_WORK_DIR)
            DEFAULT_SIM_WORK_DIR.mkdir(parents=True, exist_ok=True)
            yield
        finally: DEFAULT_SIM_WORK_DIR=old

def get_default_sim_work_dir() -> Path:
    return DEFAULT_SIM_WORK_DIR

def unload_my_imports(imports=['hemtthermal']):
    modules_to_drop=[k for k in sys.modules if any((i in k for i in imports))]
    if len(modules_to_drop):
        print(f"Unloading {', '.join(sorted(modules_to_drop))}")
    for k in modules_to_drop:
        del sys.modules[k]

um = 1.0
nm = 1.0e-3 * um
mm = 1.0e3 * um
m = 1.0e6 * um

K = 1.0
W = 1.0

@contextmanager
def time_it(message, threshold_time:float=0):
    start_time=time.time()
    yield
    took_time=time.time()-start_time
    if took_time>threshold_time:
        print(f"{message} took {took_time:.5g}s")