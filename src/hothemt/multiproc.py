from contextlib import contextmanager
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from multiprocessing import Pool
import pandas as pd
import time


from hothemt import get_default_sim_work_dir, get_log_dir, get_summary_path, specific_sim_work_dir
from hothemt.hemt import HEMT


@dataclass(eq=False)
class Task:
    def run(self) -> dict[str,Any]:
        raise NotImplementedError()
    def as_record(self) -> dict[str,Any]:
        raise NotImplementedError()
    def summary_group(self) -> str:
        raise NotImplementedError()
    def taskid(self) -> str:
        # Should be overridden by subclasses for unique string id
        raise NotImplementedError()
    def __hash__(self):
        return hash(self.taskid())
    def __eq__(self,other:'Task'):
        return self.taskid()==other.taskid()
    
    @contextmanager
    def _redirect_log(self):
        old_stdout=sys.stdout
        old_stderr=sys.stderr
        with open(get_log_dir()/f"{self.summary_group()}_{self.taskid()}.log", 'a') as logf:
            sys.stdout=logf
            sys.stderr=logf
            try:
                print("###################### LOG START ########################")
                yield
                print("###################### LOG END ##########################")
            finally:
                sys.stdout=old_stdout
                sys.stderr=old_stderr
            
    def __call__(self, sim_work_dir: Optional[Path|str]) -> dict[str,Any]:
        with specific_sim_work_dir(sim_work_dir):
            with self._redirect_log():
                return self.run()


def read_status_table(summary_group:str, sim_work_dir: Optional[str|Path]=None) -> pd.DataFrame|None:
    with specific_sim_work_dir(sim_work_dir):
        return pd.read_csv(get_summary_path(summary_group),index_col='taskid')\
                if get_summary_path(summary_group).exists() else None 
def read_status_table_as_dict(summary_group:str, sim_work_dir: Optional[str|Path]=None) -> dict[str,dict[str,Any]]:
    df = read_status_table(summary_group,sim_work_dir)
    return df.to_dict(orient='index') if df is not None else {} # type: ignore

def run_tasks(task_dependencies: dict[Task,list[Task]], nproc:Optional[int]=None):
    if nproc is None:
        from os import process_cpu_count
        nproc=int((process_cpu_count() or 4)/2)

    need_to_start_tasks=set(task_dependencies.keys())
    completed_tasks=set()
    failed_tasks={}
    summary_groups=set(task.summary_group() for task in need_to_start_tasks)
    status_tables:dict[str,dict[str,dict[str,Any]]]={sgrp:read_status_table_as_dict(sgrp) for sgrp in summary_groups}
    for task in need_to_start_tasks:
        status_tables[task.summary_group()][task.taskid()]={**task.as_record(), 'Status':'Pending'}

    status_table_updated=True
    def success_callback(result, task):
        nonlocal status_table_updated
        completed_tasks.add(task)
        status_tables[task.summary_group()][task.taskid()]['Status']='Completed'
        for k,v in result.items():
            status_tables[task.summary_group()][task.taskid()][k]=v
        running_tasks.remove(task)
        status_table_updated=True
        
    def error_callback(e, task):
        nonlocal status_table_updated
        print(f"Task {task} failed with exception {e}")
        failed_tasks[task]=e
        status_tables[task.summary_group()][task.taskid()]['Status']='Failed'
        running_tasks.remove(task)
        status_table_updated=True

    def update_status_csv_if_needed():
        nonlocal status_table_updated
        if status_table_updated:
            human_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            for sgrp,table in status_tables.items():
                df=pd.DataFrame.from_dict(table,orient='index')
                df.to_csv(get_summary_path(sgrp),index=True, index_label='taskid')
                df=df.loc[[td.taskid() for td in task_dependencies.keys() if td.summary_group()==sgrp]]
                print(f"Update at {human_time} for {sgrp}: {len(df[df['Status']=='Pending'])} pending/"\
                      f" {len(df[df['Status']=='Running'])} running/"\
                      f" {len(df[df['Status']=='Completed'])} completed/"\
                      f" {len(df[df['Status']=='Failed'])} failed")
        status_table_updated=False

    running_tasks=[]
    with Pool(nproc) as p:
        while len(need_to_start_tasks)+len(running_tasks):
            for task in list(need_to_start_tasks):
                dependencies=task_dependencies[task]
                if all(dep in completed_tasks for dep in dependencies):
                    need_to_start_tasks.remove(task)
                    status_tables[task.summary_group()][task.taskid()]['Status']='Running'
                    print(f"Running {task}")
                    running_tasks.append(task)
                    p.apply_async(task,kwds={'sim_work_dir': get_default_sim_work_dir()},
                          callback=lambda r, t=task: success_callback(r,t),
                          error_callback=lambda e, t=task: error_callback(e,t))
                elif any(dep in failed_tasks for dep in dependencies):
                    print(f"Skipping {task} because a dependency failed")
                    need_to_start_tasks.remove(task)
                    failed_tasks[task]="Dependency failed"
                    status_tables[task.summary_group()][task.taskid()]['Status']='Skipped'
            update_status_csv_if_needed()
            time.sleep(1)
        p.close()
        p.join()
        update_status_csv_if_needed()
        print("Complete")
        if len(failed_tasks):
            print(f"{len(failed_tasks)} tasks failed:")
            for task,e in failed_tasks.items():
                print(f"Task {task} failed with exception {e}")
    
@dataclass(eq=False)
class MeshTask(Task):
    hemt: HEMT
    force_remesh: bool = False
    retries: int = 2
    def __str__(self) -> str:
        return f"MeshTask for {self.hemt}"
    def taskid(self) -> str:
        return f'Mesh{self.hemt.meshid}'
    def as_record(self): return self.hemt.mesh_dict()
    def summary_group(self): return 'Mesh'
    def run(self):
        print("###################### LOG START ########################")
        for tryno in range(self.retries):
            print(f"Starting mesh for {self.hemt}, attempt {tryno+1}")
            try:
                self.hemt.get_mesh_path(force_remesh=self.force_remesh, show_gui=False)
            except Exception as e:
                if tryno==self.retries-1: raise e
            else:
                print(f"Finished mesh for {self.hemt} on attempt {tryno+1}")
                break
        return {}

@dataclass(eq=False)
class SimTask(Task):
    hemt: HEMT
    force_resim: bool = False
    def __str__(self) -> str:
        return f"SimTask for {self.hemt}"
    def taskid(self) -> str:
        return f'Sim{self.hemt.simid}'
    def as_record(self): return self.hemt.sim_dict()
    def summary_group(self): return 'Sim'
    def run(self):
        print("###################### LOG START ########################")
        print(f"Starting for {self.hemt}")
        info=self.hemt.get_siminfo(force_resim=self.force_resim)
        print(f"Finished")
        return {k:v for k,v in info.items() if isinstance(v,(int,float,str))}

def run_hemts(hemts: Sequence[HEMT],nproc:Optional[int]=None,
              force_remesh: bool = False, force_resim: bool = False,
              mesh_only:bool = False):
    task_dependencies: dict[Task,list[Task]]={}
    for h in hemts:
        mt=MeshTask(h,force_remesh=force_remesh)
        task_dependencies[mt]=[]
        if not mesh_only:
            st=SimTask(h,force_resim=force_resim)
            task_dependencies[st]=[mt]
    run_tasks(task_dependencies, nproc=nproc)
