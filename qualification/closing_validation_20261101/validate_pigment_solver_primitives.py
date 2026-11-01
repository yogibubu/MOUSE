"""Remote analytic-action and native d-frame derivative regression."""
from pathlib import Path
import socket,runpy,json
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from scipy.linalg import expm
from types import SimpleNamespace
from threadpoolctl import threadpool_limits
from matrix_gfn2._angular_transport import d_rotation
from matrix_gfn2.nonlocal_response import TransportedOperators
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.nosi_surface import PreparedNOSIModel
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.roks_orbital_response import orbital_residual_action,orbital_residual_jacobian
from matrix_gfn2._nosi_orthogonality import ground_overlap_action,ground_overlap_hessian
with threadpool_limits(2):
 rng=np.random.default_rng(182);anchor=rng.normal(size=(4,3));xyz=anchor+.02*rng.normal(size=anchor.shape);operators=rng.normal(size=(3,12,12));operators=.5*(operators+operators.transpose(0,2,1));symbols=['3s']+['3p']*3+['3d']*5+['1s']*3;atoms=[0]*9+[1,2,3]
 def transport(coords):return TransportedOperators(SimpleNamespace(coordinates_bohr=coords,ao_atom=atoms,ao_shell=symbols,atomic_numbers=[12,1,1,1]),anchor,operators)
 a=transport(xyz);du,d2r=a.frame_jet(second=True);v=rng.normal(size=xyz.size);v/=np.linalg.norm(v);h=2e-4;tangent=np.einsum('a,aij->ij',v,du);mixed=a.directional_second(v);adjoint=rng.normal(size=(12,12));adjoint=.5*(adjoint+adjoint.T);second=np.einsum('a,aij->ij',v,mixed);scalar=float(v@a.frame_curvature(adjoint)@v)
 estimates=[];firsts=[]
 for step in [h,h/2]:
  plus=transport(xyz+step*v.reshape(-1,3));minus=transport(xyz-step*v.reshape(-1,3));estimates.append((plus.u+minus.u-2*a.u)/step**2);firsts.append((plus.u-minus.u)/(2*step))
 first_error=float(np.max(abs((4*firsts[1]-firsts[0])/3-tangent)));second_error=float(np.max(abs((4*estimates[1]-estimates[0])/3-second)));contraction_error=abs(scalar-float(np.sum(second*adjoint)))
 assert first_error<1e-8 and second_error<2e-6 and contraction_error<1e-10
 k=rng.normal(size=(3,3));r=expm(k-k.T);x=rng.normal(size=(100,3))
 def harmonic(x):
  a,b,c=x.T;return np.column_stack([(2*c*c-a*a-b*b)/np.sqrt(6),np.sqrt(2)*a*c,np.sqrt(2)*b*c,(a*a-b*b)/np.sqrt(2),np.sqrt(2)*a*b])
 harmonic_error=float(np.max(abs(harmonic(x@r)-harmonic(x)@d_rotation(r))));assert harmonic_error<1e-12
 model=PreparedNOSIModel(HERE/'disentangled_bath_spectral_block_reference_eta_0.600000/closing_blind/Ethylene/model.json');s=model.anchor;c=model.arrays['states'][0];no=s.electron_count//2;n=len(c);calc=GFN2Calculator(['C','C','H','H','H','H'],temperature_kelvin=0.)
 # Use the verified actual atomic order rather than assuming XYZ order.
 calc=GFN2Calculator([{1:'H',6:'C'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);ctx=prepare_gfn2_deltascf(calc,s.coordinates_bohr/ANGSTROM_TO_BOHR,orbital_reference=s,spin_reference=model.response);occ=np.zeros((2,n));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1;left,right=np.triu_indices(n,1);triplet=np.array([np.maximum(*occ),np.minimum(*occ)]);active=np.any(np.vstack([occ,triplet])[:,left]!=np.vstack([occ,triplet])[:,right],axis=0);left,right=left[active],right[active];errors=[]
 for occupations in [occ,triplet]:
  density=np.array([(c*o)@c.T for o in occupations]);f,_=ctx.fock_and_energy(density);state=SimpleNamespace(**vars(ctx.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=occupations;kernel=ctx.density_response_kernel(state);matrix=orbital_residual_jacobian(c,occupations,f,kernel,ctx.polarization,left,right)
  for repeat in range(3):
   vector=rng.normal(size=len(left));vector/=np.linalg.norm(vector);action=orbital_residual_action(c,occupations,f,kernel,ctx.polarization,left,right,vector);errors.append(float(np.max(abs(action-matrix@vector))));symmetric_action=orbital_residual_action(c,occupations,f,kernel,ctx.polarization,left,right,vector,symmetric=True);errors.append(float(np.max(abs(symmetric_action-.5*(matrix+matrix.T)@vector))))
 ground=model.arrays['ground'];hc=ground_overlap_hessian(ground,c,s.overlap,occ,left,right);vector=rng.normal(size=len(left));vector/=np.linalg.norm(vector);constraint_error=float(np.max(abs(ground_overlap_action(ground,c,s.overlap,occ,left,right,vector)[1]-hc@vector)));assert constraint_error<1e-10
 assert max(errors)<1e-10
 report=dict(host=socket.getfqdn(),d_harmonic_rotation_error=harmonic_error,d_frame_first_error=first_error,d_frame_second_error=second_error,d_frame_curvature_contraction_error=contraction_error,orbital_Jacobian_action_max_error=max(errors),ground_constraint_action_max_error=constraint_error,finite_differences='frame derivative validation only; no production differences')
 (HERE/'pigment_solver_primitives.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
