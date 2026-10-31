"""Extend assignments using completed corrected-GFN2 TDA spectra, without rerunning them."""
import socket,sys,os,json,hashlib,shutil
from pathlib import Path
assert socket.getfqdn()=='oracle.sns.it'
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');R=Path('/home/enzo/MOUSE-state-following-20261031');D=R/'reduced_validation'
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MOUSE_VALIDATION_ROOT=str(V))
sys.path.insert(0,str(V/'source/qualification'));import remote_bootstrap
import numpy as np
from matrix_gfn2.deltascf_derivatives import load_derivative_point
name='Phthalimide';out=D/'tda'/name;case=json.loads((D/'prepared'/name/'case.json').read_text());targets=case['targets'];gp=D/'prepared'/name/'ground.npz'
with np.load(gp) as f:xyz=f['coordinates_angstrom'].copy();req=str(f['request_hash'])
s=load_derivative_point(gp,request_hash=req,coordinates_angstrom=xyz,require_gradient=False).reference_state
oldoi=np.flatnonzero(s.alpha_occupations==1);oldvi=np.flatnonzero(s.alpha_occupations==0);profiles=[];records=[]
for i in range(2):
 report_path=out/str(i)/'result.json';report=json.loads(report_path.read_text());cache=out/str(i)/'spectrum.npz';sha=hashlib.sha256(cache.read_bytes()).hexdigest()
 with np.load(cache) as f:c=f['coefficients'].copy();values=f['energies_hartree'].copy();vectors=f['vectors'].copy();matrix=f['matrix'].copy()
 occ=np.arange(len(oldoi));virt=np.arange(len(oldoi),len(c));existing={r['state'] for r in report['rows']}
 for target in targets:
  if target['id'] in existing:continue
  with np.load(target['transition_file']) as f:amplitude=f['amplitude'].copy()
  projected=(c[:,occ].T@s.overlap@s.coefficients[:,oldoi])@amplitude@(s.coefficients[:,oldvi].T@s.overlap@c[:,virt]);norm=float(np.linalg.norm(projected));total_norm=norm*target['reference_transition_norm']
  scores=abs(projected.ravel()@vectors)/max(norm,1e-300);k=int(np.argmax(scores));margin=float(scores[k]-np.max(np.delete(scores,k),initial=0.));qualified=bool(total_norm>=.7 and scores[k]>=.7 and margin>=.05 and values[k]>0)
  residual=float(np.max(abs(matrix@vectors[:,k]-values[k]*vectors[:,k])));assert residual<1e-8
  row=dict(eigen_residual_hartree=residual,state=target['id'],status='matched' if qualified else 'unassigned',excitation_ev=float(values[k]*27.211386245988),root=k+1,overlap=float(scores[k]),margin=margin,projection_norm=total_norm)
  report['rows'].append(row);records.append(dict(profile=i,row=row,spectrum_sha256=sha,spectrum_recomputed=False))
 if records:
  backup=report_path.with_name('result_before_seed_recovery.json')
  if not backup.exists():shutil.copy2(report_path,backup)
  report['assignment_extension']=dict(spectrum_sha256=sha,spectrum_recomputed=False,source=str(V/'complete_cached_tda_assignments.py'))
  report_path.write_text(json.dumps(report,indent=2)+'\n')
 assert hashlib.sha256(cache.read_bytes()).hexdigest()==sha
 profiles.append(report)
final=out/'result.json';old=json.loads(final.read_text());backup=out/'result_before_seed_recovery.json'
if not backup.exists():shutil.copy2(final,backup)
rows=[]
for target in targets:
 parents=[next(r for r in p['rows'] if r['state']==target['id']) for p in profiles];qualified=all(p['status']=='matched' for p in parents)
 rows.append(dict(state=target['id'],status='matched' if qualified else 'unassigned',parent_energies_ev=[p['excitation_ev'] for p in parents],mean_parent_tda_ev=float(np.mean([p['excitation_ev'] for p in parents])) if qualified else None))
old.update(rows=rows,profiles=profiles);final.write_text(json.dumps(old,indent=2)+'\n')
(V/'tda_assignment_recovery.json').write_text(json.dumps(dict(host=socket.getfqdn(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=records,reference_labels_read=False),indent=2)+'\n')
print(json.dumps(rows,indent=2))
