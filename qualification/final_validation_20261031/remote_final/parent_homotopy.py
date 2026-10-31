"""Follow the same CSF from F to Q through the one-body operator, no refit."""
from common import *
from matrix_gfn2.multicsf import MultiCSFObjective

def follow_parent(calc,xyz,s,responses,eri,no,rank,c0,q,seed,out):
 from independent import optimize_parent
 out=Path(out);out.mkdir(parents=True,exist_ok=True);f,qq=responses
 for key in ['operators','pair_charge_hartree','pair_spin_hartree','pair_reference_moments']:
  assert np.array_equal(getattr(f,key),getattr(qq,key)),'Homotopy requires identical nonlinear response'
 meta=dict(fingerprint_F=f.fingerprint,fingerprint_Q=qq.fingerprint,rank=rank,source_sha256=sha(__file__),energy_labels_used=False)
 if (out/'metadata.json').exists():assert json.loads((out/'metadata.json').read_text())==meta
 else:write(out/'metadata.json',meta)
 current=0.;step=.125;c=seed.copy();accepted=out/'accepted.npz'
 if accepted.exists():
  with np.load(accepted) as a:current=float(a['t']);c=a['coefficients'].copy()
 trace=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
 for iteration in range(64):
  if current>=1-1e-12:return c
  target=min(1.,current+step);response=replace(qq,pair_linear_hartree=(1-target)*f.pair_linear_hartree+target*qq.pair_linear_hartree)
  context=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=response);obj=MultiCSFObjective(context,eri,no,rank,c0,q)
  try:
   trial=optimize_parent(obj,c,out/f't_{target:.7f}')
   e,g,h=obj.evaluate(trial,True);assert np.max(abs(g))<1e-9 and obj.fidelity(trial)[0]**2>.5 and obj.cache['coefficient_margin']>=.05
  except Exception:
   trace.append(dict(t_from=current,t_target=target,status='subdivide',step=step));write(out/'history.json',trace);step*=.5
   if step<1/256:raise RuntimeError('Parent homotopy reached its minimum operator increment; checkpoints retained')
   continue
  c=trial;current=target;np.savez_compressed(out/'accepted.tmp.npz',t=current,coefficients=c);(out/'accepted.tmp.npz').replace(accepted)
  trace.append(dict(t=current,status='accepted',residual=float(np.max(abs(g))),fidelity=obj.fidelity(c)[0],observed_index=int(np.count_nonzero(np.linalg.eigvalsh(h)<-1e-7)),response_fingerprint=response.fingerprint));write(out/'history.json',trace)
 raise RuntimeError('Parent homotopy step budget exhausted; accepted checkpoint retained')
