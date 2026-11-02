"""Archive complete certificates and their unchanged-model continuation history."""
from pathlib import Path
import argparse,ast,hashlib,json,re,socket,shutil
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
p=argparse.ArgumentParser();p.add_argument('--allow-incomplete',action='store_true');args=p.parse_args()
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];out=HERE/'pigment_stagnation_repair_20261102';before=out/'before';folder=HERE/'pigment_roks/Chlorophyllide_a';out.mkdir(exist_ok=True)
phases={1:['state','aux_state','retracted_state','moving_state','complete_krylov_state','eigenvector_state','corrector_state','tangent_state','svd_full_state'],2:['state','aux_state','retracted_state','moving_state','complete_krylov_state','eigenvector_state','corrector_state','tangent_state','svd_full_state'],3:['aux_state','retracted_state','resumed_state','complete_krylov_state','eigenvector_state','corrector_state','tangent_state','svd_full_state'],4:[]}
rows=[]
for state in range(1,5):
 seed=json.loads((before/f'state_{state}/progress.json').read_text())['row'];curve=[seed];logs=[]
 for phase in phases[state]:
  path=HERE/f'chlorophyllide_repair_{phase}_{state}.log'
  if not path.exists():continue
  logs.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  destination=out/'logs'/path.with_suffix('.txt').name;destination.parent.mkdir(exist_ok=True);shutil.copy2(path,destination)
  for line in path.read_text().splitlines():
   if not line.startswith(f'Chlorophyllide_a {state} '):continue
   text=re.sub(r'np\.float64\(([^()]*)\)',r'\1',line.split(' ',2)[2])
   try:row=ast.literal_eval(text)
   except (ValueError,SyntaxError):continue
   if not isinstance(row,dict) or 'residual' not in row:continue
   if abs(row['residual']-curve[-1]['residual'])<1e-12 and abs(row['energy_hartree']-curve[-1]['energy_hartree'])<1e-11:continue
   curve.append(row)
 certificate=folder/f'state_{state}/certificate.json';result=json.loads(certificate.read_text()) if certificate.exists() else None
 if result is not None:
  for entry in result.get('energy_escape_history',result['report']['history']):
   row=dict(entry);ov=float(row.get('ground_overlap',0.));row['residual']=max(row['residual'],abs(ov));row['residual_l2']=float((row.get('residual_l2',row['residual'])**2+ov**2)**.5)
   if abs(row['residual']-curve[-1]['residual'])>1e-12 or abs(row['energy_hartree']-curve[-1]['energy_hartree'])>1e-11:curve.append(row)
 target=out/'after'/f'state_{state}';target.mkdir(parents=True,exist_ok=True)
 for name in ['checkpoint.npz','progress.json','certificate.json']:
  path=folder/f'state_{state}'/name
  if path.exists():shutil.copy2(path,target/name)
 unchanged=hashlib.sha256((target/'checkpoint.npz').read_bytes()).hexdigest()==hashlib.sha256((before/f'state_{state}/checkpoint.npz').read_bytes()).hexdigest()
 rows.append(dict(state=state,converged=result is not None and result['report']['converged'],before=seed,curve=curve,certificate=result,checkpoint_identical_to_before=unchanged,logs=logs))
complete=all(row['converged'] for row in rows)
identity_audit=json.loads((out/'stationarity_audit.json').read_text()) if (out/'stationarity_audit.json').exists() else None
complete=complete and identity_audit is not None and identity_audit['complete']
if not complete and not args.allow_incomplete:raise RuntimeError('Production targets are not all certified; no completion report written')
assert rows[3]['checkpoint_identical_to_before'],'Previously certified target 4 was modified'
sources={}
for path in [BASE/'source/packages/matrix-gfn2/src/matrix_gfn2/_roks_matrix_free.py',HERE/'run_pigment_roks.py',HERE/'run_pigment_energy_escape.py',HERE/'finalize_pigment_roks.py']:
 sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
report=dict(host=socket.getfqdn(),complete=complete,identity_audit=identity_audit,tolerance=1e-10,states=rows,source_sha256=sources,model_sha256=hashlib.sha256((HERE/'pigment_models/Chlorophyllide_a/manifest.json').read_bytes()).hexdigest(),scope='Full functional ROKS orbital stationarity; final screened-CID spectrum and pigment nuclear Hessian are separate qualifications')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(complete=complete,states=[dict(state=r['state'],converged=r['converged'],residual=r['certificate']['report']['residual'] if r['certificate'] else r['curve'][-1]['residual']) for r in rows]),indent=2))
