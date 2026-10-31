"""Frozen-anchor geometry model for analytic derivative qualification only."""
from pathlib import Path
import os,sys,socket,json,hashlib
assert socket.getfqdn()=='oracle.sns.it'
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');D=Path('/home/enzo/MOUSE-state-following-20261031/reduced_validation');P=Path('/home/enzo/MOUSE-publication-validation-20261030')
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MOUSE_VALIDATION_ROOT=str(V));sys.path.insert(0,str(V/'source/qualification'));import remote_bootstrap
import numpy as np
from matrix_gfn2 import GFN2Calculator,FrozenNonlocalResponseReference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from multicsf_objective import MultiCSFObjective
from multicsf_nuclear_response import FrozenLowdinERI,metric_power

class ParentGeometryModel:
    def __init__(self,name,sid,index,attempt):
        self.name=name;self.sid=sid;self.index=index
        cp=D/'prepared'/name/'ground.npz'
        with np.load(cp) as a:self.xyz=a['coordinates_angstrom'].copy();req=str(a['request_hash'])
        self.anchor=load_derivative_point(cp,request_hash=req,coordinates_angstrom=self.xyz,require_gradient=False).reference_state
        self.symbols=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in self.anchor.atomic_numbers]
        calc=GFN2Calculator(self.symbols,temperature_kelvin=0.);native=prepare_gfn2_deltascf(calc,self.xyz,orbital_reference=self.anchor)
        path=P/'results/full_response'/name/'model.npz' if index==0 else D/'models'/name/'model.npz'
        with np.load(path) as a:data={k:a[k].copy() for k in ['operators','charge','spin','q0','linear']}
        self.ref=FrozenNonlocalResponseReference.from_vertical_model(self.anchor,native.kernel.partition,data,hashlib.sha256(path.read_bytes()).hexdigest())
        with np.load(P/'results/full_response'/name/'projected_eri.npz') as a:eri=a['eri'].copy()
        self.eri=FrozenLowdinERI(self.anchor,eri)
        with np.load(D/'tda'/name/str(index)/'spectrum.npz') as a:self.reference_c=a['coefficients'].copy()
        case=json.loads((D/'prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']==sid)
        with np.load(target['transition_file']) as a:amp=a['amplitude'].copy();oi=a['occupied'].copy();vi=a['virtual'].copy()
        self.no=len(oi);self.projected=self.reference_c[:,:self.no].T@self.anchor.overlap@self.anchor.coefficients[:,oi]@amp@self.anchor.coefficients[:,vi].T@self.anchor.overlap@self.reference_c[:,self.no:];self.projected/=np.linalg.norm(self.projected)
        out=V/'pi_pi_repair/optimized'/name/sid/attempt;report=json.loads((out/'result.json').read_text());self.rank=report['rank'];assert report['status']=='stationary_candidate'
        cp=out/('latest_released.npz' if '_fr_' in attempt else 'latest.npz')
        with np.load(cp) as a:self.coefficients=a['coefficients'].copy()
        self.anchor_root=metric_power(self.anchor.overlap,.5)
    def objective(self,xyz):
        calc=GFN2Calculator(self.symbols,temperature_kelvin=0.);ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=self.anchor,spin_reference=self.ref)
        t,l=self.eri.chart(ctx.state);map=metric_power(ctx.state.overlap,-.5)@t.u@self.anchor_root
        cr=map@self.reference_c;initial=map@self.coefficients
        obj=MultiCSFObjective(ctx,self.eri.ao_eri(ctx.state),self.no,self.rank,cr,self.projected)
        return obj,initial
