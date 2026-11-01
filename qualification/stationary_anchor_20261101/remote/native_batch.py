"""Exact fixed-geometry native GFN2 response in a complete AO-pair chart.

Only shell cubic and D4/ATM charge curvature depend on the reference density.
The static anisotropic/shell response is reused; no term is approximated.
"""
from anchor import *

class NativeBatch:
    def __init__(self,raw,b,kbase,shell,atom,delta_shell,delta_d4):
        self.raw=raw;self.shape=raw.overlap.shape;self.b=b;self.kbase=kbase
        self.shell=shell;self.atom=atom;self.delta_shell=delta_shell;self.delta_d4=delta_d4
    def __getattr__(self,key):return getattr(self.raw,key)
    def __call__(self,dp):return self.response_batch(np.asarray(dp)[None])[0]
    def response_batch(self,dp):
        dp=np.asarray(dp)
        if dp.ndim!=3 or dp.shape[1:]!=self.shape or not np.all(np.isfinite(dp)):raise ValueError('invalid dense density-response batch')
        if np.max(abs(dp-dp.transpose(0,2,1)),initial=0.)>2e-12:raise ValueError('density perturbations must be symmetric')
        flat=dp.reshape(len(dp),-1);moments=flat@self.b.T
        result=(moments@self.kbase.T)@self.b
        result+=((flat@self.shell.T)*self.delta_shell)@self.shell
        result+=((flat@self.atom.T)@self.delta_d4.T)@self.atom
        return result.reshape(dp.shape)

def install_native_batch(case,out):
    calc=case['calc'];s=case['s'];original=calc.density_response_kernel
    assert getattr(calc,'_active_nn_state',None) is None,'This qualification covers ordinary GFN2 only'
    with np.load(out/'embedded_ground.npz') as a:c=a['coefficients'].copy()
    state=SimpleNamespace(**vars(s));state.coefficients=c;base=original(state)
    with np.load(out/'F/model.npz') as a:operators=a['operators'].copy();correction=a['charge'].copy()
    b=.5*(s.overlap@operators+operators@s.overlap);b=b.reshape(len(b),-1)
    native_path=out/'native_response.npz'
    if native_path.exists():
        with np.load(native_path) as a:kbase=a['matrix'].copy()
    else:
        with np.load(out/'response_progress.npz') as a:full=a['charge'].copy()
        kbase=.5*(full+full.T)-correction;np.savez_compressed(native_path,matrix=kbase)
    shell_index=base.partition.orbital_shell;atom_index=base.operators.atom_index
    # Boolean addition must count both equal endpoints rather than perform OR.
    shell=np.array([-.5*s.overlap*((np.asarray(shell_index)==i).astype(float)[:,None]+(np.asarray(shell_index)==i).astype(float)[None,:]) for i in range(base.partition.nshells)])
    atom=np.array([-.5*s.overlap*((np.asarray(atom_index)==i).astype(float)[:,None]+(np.asarray(atom_index)==i).astype(float)[None,:]) for i in range(len(s.atomic_numbers))])
    shell=shell.reshape(len(shell),-1);atom=atom.reshape(len(atom),-1)
    base_d4=base._d4_charge_response_hessian.copy()
    if base.atm_charge_hessian is not None:base_d4+=base.atm_charge_hessian
    def factory(state):
        if not np.array_equal(state.coordinates_bohr,s.coordinates_bohr):raise ValueError('fixed-geometry native response chart cannot move nuclei')
        raw=original(state);delta=raw._d4_charge_response_hessian-base_d4
        if raw.atm_charge_hessian is not None:delta=delta+raw.atm_charge_hessian
        delta_shell=2*np.asarray(raw.partition.third_order_hartree)*(raw.reference_shell_charges-base.reference_shell_charges)
        return NativeBatch(raw,b,kbase,shell,atom,delta_shell,delta)
    calc.density_response_kernel=factory
    return original,factory
