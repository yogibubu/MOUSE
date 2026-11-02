"""Independent directional derivative audit at the production checkpoint."""
import socket,json,time
from pathlib import Path
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
from run_pigment_roks import load,HERE
import numpy as np
from scipy.linalg import expm
from types import SimpleNamespace
from matrix_gfn2.roks_orbital_response import roks_residual_action
from matrix_gfn2._nosi_orthogonality import ground_overlap_jet,prepare_ground_overlap_action
from threadpoolctl import threadpool_limits
with threadpool_limits(8):
 model,state,reference,calc,base,full=load('Chlorophyllide_a');no=state.electron_count//2;n=len(state.coefficients)
 with np.load(HERE/'pigment_roks/Chlorophyllide_a/state_2/checkpoint.npz') as z:c=z['coefficients'].copy();mu=float(z['multiplier'])
 occ=np.zeros((2,n));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1;triplet=np.array([np.maximum(*occ),np.minimum(*occ)]);left,right=np.triu_indices(n,1);keep=np.any(np.vstack([occ,triplet])[:,left]!=np.vstack([occ,triplet])[:,right],axis=0);left,right=left[keep],right[keep];s=state.overlap;ground=state.coefficients
 def evaluate(context,coefficient):
  components=[];g=np.zeros(len(left));energy=0.
  for weight,o in [(2.,occ),(-1.,triplet)]:
   density=np.array([(coefficient*x)@coefficient.T for x in o]);f,e=context.fock_and_energy(density);components.append((weight,o,f));energy+=weight*e;q=sum((coefficient.T@ff@coefficient)*x[None,:] for ff,x in zip(f,o));g+=2*weight*(q[left,right]-q[right,left])
  overlap,j=ground_overlap_jet(ground,coefficient,s,occ,left,right)
  return energy,g-mu*j,components,j,overlap
 e,r,components,j,overlap=evaluate(full,c);v=r/np.linalg.norm(r);rotation=np.zeros((n,n));rotation[left,right]=v;rotation[right,left]=-v;dc=c@rotation
 kernel_state=SimpleNamespace(**vars(state));kernel_state.coefficients=c;kernel_state.alpha_occupations,kernel_state.beta_occupations=occ
 rows=[]
 for name,context in [('native_RI',base),('full',full)]:
  energy,residual,parts,jj,ov=evaluate(context,c);kernel=context.density_response_kernel(kernel_state);constraint=prepare_ground_overlap_action(ground,c,s,occ,left,right);analytic=roks_residual_action(c,parts,kernel,context.polarization,left,right,v)-mu*constraint(v)[1]
  # v equals the full constrained gradient, so its moving-frame connection vanishes for the full model.
  qmatrix=np.zeros((n,n));qmatrix[left,right]=residual;qmatrix[right,left]=-residual;connection=.5*(rotation@qmatrix-qmatrix@rotation)[left,right];moving=analytic-connection
  for h in [1e-4,2e-5]:
   ep,rp,fp,jp,op=evaluate(context,c@expm(h*rotation));em,rm,fm,jm,om=evaluate(context,c@expm(-h*rotation));fd=(rp-rm)/(2*h);fock_errors=[]
   for (weight,o,f),plus,minus in zip(parts,fp,fm):
    dp=np.array([(dc*x)@c.T for x in o]);dp+=dp.transpose(0,2,1);pred=kernel(dp.sum(axis=0))[None]+context.polarization.response_batch(dp[None])[0];actual=(plus[2]-minus[2])/(2*h);fock_errors.append(dict(max_error=float(np.max(abs(pred-actual))),relative_error=float(np.linalg.norm(pred-actual)/np.linalg.norm(actual))))
   row=dict(context=name,h=h,residual_norm=float(np.linalg.norm(residual)),energy_derivative_error=float((ep-em)/(2*h)-(residual+mu*jj)@v),Jacobian_relative_error=float(np.linalg.norm(fd-moving)/np.linalg.norm(fd)),Jacobian_max_error=float(np.max(abs(fd-moving))),merit_derivative_actual=float(residual@fd),merit_derivative_predicted=float(residual@moving),fock_errors=fock_errors);rows.append(row);print(json.dumps(row),flush=True)
 (HERE/'pigment_roks_derivative_probe.json').write_text(json.dumps(dict(host=socket.getfqdn(),rows=rows),indent=2)+'\n')
