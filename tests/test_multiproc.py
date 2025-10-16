from hothemt.persistent_id import DBBackedIDMixin
import pytest
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import pandas as pd

from hothemt import get_default_sim_work_dir, get_summary_path, specific_sim_work_dir
from hothemt.hemt import HEMT
from hothemt.multiproc import Task, run_hemts, run_tasks


@dataclass(eq=False,unsafe_hash=False)
class NullTask(Task):
    msg: str
    fail: bool = False
    def taskid(self) -> str:
        return "Task"+self.msg
    def run(self):
        print(f"Starting {self.msg}")
        time.sleep(1)
        if self.fail:
            print("Failing as requested")
            raise RuntimeError(f"Task {self.msg} failed as requested")
        print(f"Finished {self.msg}")
        return {'result':'Result'+self.msg}
    def as_record(self) -> dict[str, Any]:
        return {'message':self.msg}
    def summary_group(self): return 'Null'
    
def test_multiproc_null():
    swd=Path("output/output_test/multiproc_null")    
    with specific_sim_work_dir(swd, clear=True):
        print(get_default_sim_work_dir())
        run_tasks({
            NullTask("F"):[],
            NullTask("G",fail=True):[],
            NullTask("A",fail=True):[],
        })
        results=pd.read_csv(get_summary_path('Null')).drop(columns=['taskid'])
        print(results.set_index('message').to_dict(orient='index'))
        assert results.set_index('message')[['Status']].to_dict(orient='index') == \
            {'F': {'Status': 'Completed'},
             'G': {'Status': 'Failed'},
             'A': {'Status': 'Failed'}}
        run_tasks({
            NullTask("B"):[NullTask("A")],
            NullTask("A"):[],
            NullTask("D"):[NullTask("A"),NullTask("C")],
            NullTask("C",fail=True):[NullTask("B")],
            NullTask("E"):[],
        })
        results=pd.read_csv(get_summary_path('Null')).drop(columns=['taskid'])
        print(results.set_index('message').to_dict(orient='index'))
        assert results.set_index('message')[['Status']].to_dict(orient='index') == \
            {'F': {'Status': 'Completed'},
             'G': {'Status': 'Failed'},
             'A': {'Status': 'Completed'},
             'B': {'Status': 'Completed'},
             'C': {'Status': 'Failed'},
             'D': {'Status': 'Skipped'},
             'E': {'Status': 'Completed'}}

def test_multiproc_hemt():

    redo=True
    with specific_sim_work_dir("output/output_test/multiproc_hemt", clear=redo):
        from hothemt import um, W, mm, K, m
        common: dict[str,Any]= dict(
            w_f=1*um, rows=1, row_pitch=7*um, n_f=1,
            lgs=.5*um, lg=.15*um,lgd=.5*um,lc=.5*um,
            L_h=.3*um, L_ho=0*um,
            t_GaN=.5*um, t_Rel=.5*um, l_chip=3*um,
            alpha_mesh=.1,
        )
        run_hemts([
            (h1:=HEMT(**common, P_per_W=1*W/mm, T_A=40+273.15)),
            (h2:=HEMT(**common, P_per_W=2*W/mm, T_A=40+273.15))
        ],force_remesh=redo, force_resim=redo)

    with pytest.raises(AssertionError):
        # Should fail because sim_work_dir is not the same as when the hemt was run
        h1.get_siminfo()

    with specific_sim_work_dir("output/output_test/multiproc_hemt", clear=False):
        s1=h1.get_siminfo()
        s2=h2.get_siminfo()
        import numpy as np
        assert np.isclose(s1['heat_flux_in']/h1.w_f,1*W/mm,rtol=1e-3)
        assert np.isclose(s2['heat_flux_in']/h2.w_f,2*W/mm,rtol=1e-3)

    with specific_sim_work_dir("output/output_test/multiproc_hemt", clear=False):
        s1=DBBackedIDMixin.from_simid(h1.simid).get_siminfo()
        s2=DBBackedIDMixin.from_simid(h2.simid).get_siminfo()
        import numpy as np
        assert np.isclose(s1['heat_flux_in']/h1.w_f,1*W/mm,rtol=1e-3)
        assert np.isclose(s2['heat_flux_in']/h2.w_f,2*W/mm,rtol=1e-3)
        assert HEMT(**common, P_per_W=1*W/mm, T_A=40+273.15).simid==h1.simid
        assert HEMT(**common, P_per_W=2*W/mm, T_A=40+273.15).simid==h2.simid

if __name__ == '__main__':
    #test_multiproc_null()
    test_multiproc_hemt()