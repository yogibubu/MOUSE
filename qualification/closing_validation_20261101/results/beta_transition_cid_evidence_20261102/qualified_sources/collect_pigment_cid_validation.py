"""Archive compact generic pigment spectral evidence; leave large prepared arrays on compute host."""
from pathlib import Path
import socket,json,hashlib,shutil,argparse
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
protocol=json.loads((HERE/args.protocol).read_text());NAME=protocol['molecule'];prefix=protocol['artifact_prefix'];fixed_w=protocol['fixed_screening_frontier'];baseline_w=protocol['baseline_screening_frontier'];out=HERE/args.output;out.mkdir(exist_ok=True)
def copy(src,dst):
 src=Path(src);dst=out/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
assessment=json.loads((HERE/f'{prefix}_assessment.json').read_text());ns=assessment['convergence']['windows'];inventory=[]
for name in [args.protocol,f'{prefix}_convergence.json',f'{prefix}_assessment.json','pigment_spectral_references.json'] :copy(HERE/name,name)
for script in ['complete_pigment_cid_validation.py','audit_pigment_cid_validation.py','collect_pigment_cid_validation.py','prepare_pigment_screening.py','run_pigment_cid.py','run_pigment_roks.py','pigment_block_matmul.py','pigment_reference_embedding.py','pigment_compact_reference.py','qualify_pigment_rescreen.py','validate_cid_contraction_batch.py','test_slater_contraction_batch.py','test_pigment_phase.py','build_transition_screening.py','summarize_transition_spectra.py','test_transition_projector.py','complete_transition_validation.py']:copy(HERE/script,'qualified_sources/'+script)
for p in (BASE/'source/packages/matrix-gfn2/src/matrix_gfn2').glob('*.py'):
 copy(p,'qualified_sources/matrix_gfn2/'+p.name)
model=HERE/f'pigment_models/{NAME}';copy(model/'manifest.json','frozen_model_manifest.json');copy(model/'quadrature/manifest.json','frozen_quadrature_manifest.json')
for i in assessment['convergence']['protocol'].get('ROKS_ordinals',[x['ordinal'] for x in json.loads((HERE/f'pigment_roks/{NAME}/result.json').read_text())['states']]):
 for name in ['certificate.json','checkpoint.npz']:copy(HERE/f'pigment_roks/{NAME}/state_{i}/{name}',f'roks/state_{i}/{name}')
def case_mode(n,w):return protocol.get('case_modes',{}).get(f'N{n}_W{w}',protocol.get('screening_mode','legacy_sectors'))
folders=[(n,baseline_w) for n in protocol['baseline_frontiers']]+[(n,n) for n in ns]+[(n,fixed_w) for n in ns]
for n,w in dict.fromkeys(folders):
 f=HERE/(f'pigment_cid/{NAME}' if n==2 and w==5 else f'pigment_cid_expansion/{NAME}/'+('' if w==5 else f'screening_{w}/')+f'frontier_{n}')
 if case_mode(n,w)=='coherent_transitions':f=HERE/f'pigment_cid_transition/{NAME}/screening_{w}/frontier_{n}'
 dst=f'spectra/N{n}_W{w}'
 copy(f/'result.json',dst+'/result.json')
 if (f/'scalar_elements.json').exists():copy(f/'scalar_elements.json',dst+'/scalar_elements.json')
 else:
  expected=protocol.get('certified_baseline_matrix_sha256')
  if n not in protocol['baseline_frontiers'] or expected is None or digest(f/'matrices.npz')!=expected:
   raise ValueError('missing scalar journal without an independently certified baseline matrix')
  (out/dst/'legacy_scalar_journal.json').write_text(json.dumps(dict(scalar_journal_available=False,independently_certified_matrix_sha256=expected,scope='Legacy baseline; matrix identity and independently replayed matrix audit replace unavailable construction journal'),indent=2)+'\n')
 with np.load(f/'eigenstates.npz') as a:np.savez_compressed(out/dst/'eigenvectors.npz',energies=a['energies'],ci=a['ci'],transition_dipoles=a['transition_dipoles'])
 copy(f/'observables_phase_corrected.npz',dst+'/observables_phase_corrected.npz')
 p=f/'matrices.npz';inventory.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p),retained_on_remote=True))
 if n==2:copy(p,dst+'/matrices.npz')
for w,mode in sorted(set((w,case_mode(n,w)) for n,w in folders)):
 f=HERE/(f'pigment_screening/{NAME}' if w==5 else f'pigment_screening_expansion/{NAME}/frontier_{w}')
 if mode=='coherent_transitions':f=HERE/f'pigment_screening_transition/{NAME}/frontier_{w}'
 for name in ['certificate.json','gap_diagnostic.json']:
  if (f/name).exists():copy(f/name,f'screening/W{w}/'+name)
for p in HERE.glob(prefix+'*.log'):copy(p,'logs/'+p.stem+'.txt')
for control in protocol.get('saved_generality_controls',[]):
 for p in HERE.glob(control+'*.json'):copy(p,'generality_control/'+p.name)
 control_log=HERE/(control+'.log')
 if control_log.exists():copy(control_log,'generality_control/'+control+'.txt')
for name in protocol.get('extra_evidence_files',[]):copy(HERE/name,'controls/'+name)
(out/'large_artifacts.json').write_text(json.dumps(inventory,indent=2)+'\n')
files={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
(out/'artifact_manifest.json').write_text(json.dumps(dict(host=socket.getfqdn(),file_count=len(files),source_hash_scope='Source snapshot at evidence export; earlier exact algebra generations remain in git and reconstruction controls',sha256=files),indent=2)+'\n');print('ARCHIVED',len(files),'files',flush=True)
