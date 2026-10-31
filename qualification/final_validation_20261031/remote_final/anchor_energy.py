"""Recalibrate actual +/-2% references and evaluate at one common geometry."""
from common import *
import fcntl,argparse,traceback,resource

def run(name,sid,scale):
 base=MultiCSFModel(W/f'models/{name}/{sid}/model.json');src=W/f'anchor_rank_symmetry/{name}/{scale:.2f}'
 out=OUT/f'anchor_energy/{name}/{scale:.2f}';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
 start=time.monotonic();raw=loadpoint(src/'native.npz');s=canonical_state(raw);raw=replace(raw,reference_state=s)
 xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR;calc=calculator(s);ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
 data,erifile=calibrate(out,src,s,ctx)
 refs=[FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,a,sha(out/profile/'anchor/model.npz')) for a,profile in zip(data,['full_response','qualified_spectral_response'])]
 cert=json.loads((src/'native_certificate.json').read_text());grounds=[RestrictedGroundResult(raw,cert['minimum_curvature'],cert['index'])]
 for i,ref in enumerate(refs):grounds.append(ground(out/f'grounds/{i}',calc,xyz,s,ref))
 with np.load(src/'reference.npz') as a:dc=a['coefficients'].copy();dno=int(np.count_nonzero(a['occupations']==2))
 with np.load(src/'tda.npz') as a:dx=a['amplitudes'][0].copy();dx/=np.linalg.norm(dx)
 small,t=source_molecule(s);mol=scf.chkfile.load_mol(str(src/'backend.chk'));cross=(gto.intor_cross('int1e_ovlp',small,mol).T@t).T
 projection=s.coefficients.T@cross@dc;amp=projection[:base.no,:dno]@dx@projection[base.no:,dno:].T;amp/=np.linalg.norm(amp)
 weights=np.linalg.svd(amp,compute_uv=False)**2;weights/=weights.sum();assert int(np.floor(1/(weights@weights)+.5))==base.rank
 # Metric-polar transport preserves the original relative active frame.
 chart,_=base.eri.chart(s);transport=metric_power(s.overlap,-.5)@chart.u@metric_power(base.anchor.overlap,.5)
 initial=np.array([transport@c for c in base.arrays['initial_coefficients']]);rc=[];rt=[]
 for g in grounds[1:]:
  c=g.coefficients[0];rc.append(c);q=c[:,:base.no].T@s.overlap@s.coefficients[:,:base.no]@amp@s.coefficients[:,base.no:].T@s.overlap@c[:,base.no:];rt.append(q/np.linalg.norm(q))
 bundle=out/'portable_model'
 if not (bundle/'model.json').exists():
  with np.load(erifile) as a:eri=a['eri'].copy()
  write_multicsf_model(bundle,template=raw,responses=refs,eri=eri,initial_coefficients=initial,reference_coefficients=rc,reference_transitions=rt,transition_weights=weights,grounds=grounds,provenance=dict(anchor_source=str(src),original_model=base.fingerprint,relative_frame='metric-polar transported original full active Gram matrix',source_sha256=sha(__file__)))
 model=MultiCSFModel(bundle/'model.json');dest=out/'common_target'
 if (dest/'result.json').exists():point=load_multicsf_result(dest/'result.json',model,calc)
 else:
  point=evaluate_multicsf_surface(calc,base.xyz,model,compute_hessian=False);save_multicsf_result(dest,point,model)
 baseline=OUT/f'anchor_energy/{name}/baseline'
 if (baseline/'result.json').exists():original=load_multicsf_result(baseline/'result.json',base,calc)
 else:original=evaluate_multicsf_surface(calc,base.xyz,base,compute_hessian=False);save_multicsf_result(baseline,original,base)
 gap=.5*sum(point.parent_gaps_hartree)*EV;oldgap=.5*sum(original.parent_gaps_hartree)*EV
 row=dict(status='complete',host=HOST,molecule=name,state=sid,scale=scale,common_coordinates_angstrom=base.xyz,excitation_ev=gap,baseline_ev=oldgap,change_ev=gap-oldgap,parent_gaps_ev=np.array(point.parent_gaps_hartree)*EV,residual=point.orbital_residual,fidelity=point.fidelity,rank=model.rank,seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_fingerprint=model.fingerprint,numerical_accuracy_labels_read=False)
 write(out/'result.json',row);print(name,scale,'ENERGY CHANGE',row['change_ev'],flush=True);return row
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('sid');a.add_argument('scale',type=float);args=a.parse_args()
 lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 with threadpool_limits():run(args.name,args.sid,args.scale)
