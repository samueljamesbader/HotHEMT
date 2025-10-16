from contextlib import contextmanager
from sfepy.discrete.probes import Probe
from sfepy.base.base import Struct

@contextmanager
def probe_cache_context():
    old_probe_cache = Probe.cache
    Probe.cache = Struct(name='probe_shared_evaluate_cache')
    try:
        yield
    finally:
        Probe.cache = old_probe_cache