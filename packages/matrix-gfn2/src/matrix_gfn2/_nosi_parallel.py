"""Independent analytic nuclear columns with stationary orbitals reused.

Spawned workers own their calculator and native integral context. They do
not run SCC, SCF, TDA or an orbital search. Only the analytic matrix response
is rebuilt, using the parent's converged ground/ROKS checkpoints.
"""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np

_response=None
_roots=None
_subtract=False
_primary=None
_thread_limit=None


def _initialize(path,options,coordinates,ground,states,multipliers,roots):
    global _response,_roots,_subtract,_primary,_thread_limit
    try:
        from threadpoolctl import threadpool_limits
        _thread_limit=threadpool_limits(1)
    except ImportError:pass
    from .calculator import GFN2Calculator
    from .deltascf import prepare_gfn2_deltascf
    from .nosi_surface import PreparedNOSIModel
    from ._nosi_surface_response import MatrixResponse
    model=PreparedNOSIModel(path);calculator=GFN2Calculator(**options)
    calculator._native_thread_limit=1
    def no_scc(*args,**kwargs):raise AssertionError('new SCC forbidden in analytic NOSI worker')
    calculator.evaluate=no_scc
    context=prepare_gfn2_deltascf(calculator,coordinates,orbital_reference=model.anchor,spin_reference=model.response)
    _response=MatrixResponse(context,model.response,ground,states,multipliers,model.arrays['auxiliary_frame'],model.labels,model.arrays['h'],model.arrays['eri'],derivatives=True,auxiliary_records=model.auxiliary_records)
    _roots=np.asarray(roots,dtype=int);_subtract=model.manifest.get('static_self_energy_subtraction',False);_primary=np.arange(2*len(states))


def _column(index):
    from ._nosi_nuclear import projected_eigen_jet
    direction=np.zeros(_response.nr);direction[index]=1.
    h,s,eg=_response.matrices(direction)
    if _subtract:
        from ._nosi_static_subtraction import subtracted_matrix_jet
        h=subtracted_matrix_jet(h,s,eg,_primary)
    return index,np.column_stack([projected_eigen_jet(h,s,int(root)).ab for root in _roots])


def columns(model,calculator,coordinates,ground,states,multipliers,roots,workers):
    dimension=np.asarray(coordinates).size
    payload=(str(model.path),calculator._hessian_process_options(),np.asarray(coordinates),ground,states,multipliers,roots)
    with ProcessPoolExecutor(max_workers=min(int(workers),dimension),mp_context=get_context('spawn'),initializer=_initialize,initargs=payload) as pool:
        yield from pool.map(_column,range(dimension),chunksize=1)
