from dataclasses import dataclass, fields
from pathlib import Path
import pickle
from typing import Optional, TypeVar, Type
T=TypeVar('T',bound='DBBackedIDMixin')

@dataclass()
class DBBackedIDMixin:
    """A mixin class to provide database-backed unique IDs for instances."""

    def __post_init__(self, *args, **kwargs):
        from hothemt import get_default_sim_work_dir
        self.sim_work_dir = get_default_sim_work_dir()

    def __init_subclass__(cls,**kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(DBBackedIDMixin,'_registered_classes'):
            DBBackedIDMixin._registered_classes={}
        DBBackedIDMixin._registered_classes[cls.__name__]=cls

    @staticmethod 
    def get_class_for_sim_work_dir(sim_work_dir:Path):
        potential_dbs=list(sim_work_dir.glob("*.iddb.pkl"))
        assert len(potential_dbs)<=1, f"Multiple iddb.pkl files found in {sim_work_dir}"
        assert len(potential_dbs)==1, f"No iddb.pkl file found in {sim_work_dir}"
        dbfile=potential_dbs[0]
        clsname=dbfile.stem.split('.')[0]
        with open(dbfile,'rb') as f:
            db=pickle.load(f)
        assert clsname in DBBackedIDMixin._registered_classes, f"Class {clsname} not registered"
        assert db['class'] is DBBackedIDMixin._registered_classes[clsname], "Class mismatch, database corrupted?"
        return DBBackedIDMixin._registered_classes[clsname]

    @property
    def simid(self) -> int:
        """Return the unique simulation ID for this instance."""
        if not hasattr(self,'_simid'): self._setup_id()
        return self._simid
    
    @property
    def meshid(self) -> int:
        """Return the unique mesh ID for this instance."""
        if not hasattr(self,'_meshid'): self._setup_id()
        return self._meshid

    """A mixin class to provide database-backed unique IDs for instances."""
    @classmethod
    def from_simid(cls:Type[T], sim_id:int, sim_work_dir:Optional[str|Path]=None) -> T:
        """Load a HEMT instance from a previous simulation ID."""
        from hothemt import get_default_sim_work_dir, specific_sim_work_dir
        sim_work_dir=Path(sim_work_dir) if sim_work_dir is not None else get_default_sim_work_dir()
        cls=DBBackedIDMixin.get_class_for_sim_work_dir(sim_work_dir)
        cls._setup_iddb(sim_work_dir=sim_work_dir)
        try:
            params_tuple=next((k for k,v in cls._hemtids[sim_work_dir]['sim'].items() if v==sim_id))
        except StopIteration: raise ValueError(f"Simulation ID {sim_id} not found")
        with specific_sim_work_dir(sim_work_dir):
            h=cls(**dict(params_tuple))
        assert h.simid==sim_id, "Simulation ID mismatch, database corrupted?"
        return h
    
    # Persistent db-backed hashing functions for naming the mesh and simulation saves
    def mesh_dict(self):
        return {k.name:getattr(self,k.name) for k in fields(self)
                              if 'NotMesh' not in getattr(k.type,'__metadata__',[])}
    def mesh_tuple(self):
        """Return a tuple of parameters defining the mesh."""
        return tuple(self.mesh_dict().items())
    def sim_dict(self):
        return {k.name:getattr(self,k.name) for k in fields(self)
                              if 'NotSim' not in getattr(k.type,'__metadata__',[])}
    def sim_tuple(self):
        """Return a tuple of parameters defining the simulation."""
        return tuple(self.sim_dict().items())
    
    # Manage a unique simulation id and mesh id for a given structure,
    # backed by a local (pickle) database for persistence
    def _setup_id(self):
        cls=self.__class__
        cls._setup_iddb(self.sim_work_dir)
        needupdate=False
        if (meshid:=cls._hemtids[self.sim_work_dir]['mesh'].get(self.mesh_tuple(),None)) is None:
            self._meshid=cls._hemtids[self.sim_work_dir]['mesh'][self.mesh_tuple()]=\
                len(cls._hemtids[self.sim_work_dir]['mesh'])
            needupdate=True
        else: self._meshid=meshid
        if (simid:=cls._hemtids[self.sim_work_dir]['sim'].get(self.sim_tuple(),None)) is None:
            self._simid =cls._hemtids[self.sim_work_dir]['sim' ][self.sim_tuple() ]=\
                len(cls._hemtids[self.sim_work_dir]['sim'] )
            needupdate=True
        else: self._simid=simid
        if needupdate:
            with open(self.sim_work_dir/f"{cls.__name__}.iddb.pkl",'wb') as f:
                pickle.dump(cls._hemtids[self.sim_work_dir],f)
    @classmethod
    def _setup_iddb(cls, sim_work_dir:Path):
        if not hasattr(cls,'_hemtids'): cls._hemtids={}
        if sim_work_dir not in cls._hemtids:
            try:
                with open(sim_work_dir/f"{cls.__name__}.iddb.pkl",'rb') as f:
                    cls._hemtids[sim_work_dir]=pickle.load(f)
            except:
                cls._hemtids[sim_work_dir]={'sim':{},'mesh':{},'class':cls}
                with open(sim_work_dir/f"{cls.__name__}.iddb.pkl",'wb') as f:
                    pickle.dump(cls._hemtids[sim_work_dir],f)