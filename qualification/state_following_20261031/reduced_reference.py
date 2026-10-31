"""Molecular operator choice using only the original ground-level order check."""
from pathlib import Path
import socket,os,sys,json,hashlib,subprocess
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MOUSE_VALIDATION_ROOT='/home/enzo/MOUSE-state-following-20261031')
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=R/'reduced_validation'
sys.path.insert(0,str(R/'source/qualification'));import remote_bootstrap
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2.deltascf_derivatives import load_derivative_point
name=sys.argv[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with threadpool_limits(limits=2):
 cp=D/'prepared'/name/'ground.npz'
 with np.load(cp) as f:xyz=f['coordinates_angstrom'].copy();req=str(f['request_hash'])
 s=load_derivative_point(cp,request_hash=req,coordinates_angstrom=xyz,require_gradient=False).reference_state
 original=P/'results/qualified_spectral_response'/name/'model.npz'
 if not original.exists():original=D/'rejected_spectral'/name/'model.npz'
 with np.load(original) as f:a=f['operators'];v=f['linear'];charge=f['charge'].copy();spin=f['spin'].copy();q0=f['q0'].copy()
 potential=np.einsum('a,aij->ij',v,.5*(s.overlap@a+a@s.overlap));matrix=s.coefficients.T@(s.fock+potential)@s.coefficients
 assert np.max(abs(matrix-np.diag(np.diag(matrix))))<1e-8
 e=np.diag(matrix);gap=float(np.min(e[s.alpha_occupations==0])-np.max(e[s.alpha_occupations==1]))
 if gap>0:selected=original
 else:
  subprocess.run([sys.executable,str(R/'reduced_projected_model.py'),name],check=True)
  selected=D/'projected_reference/models'/name/'model.npz'
 with np.load(selected) as f:assert np.array_equal(f['charge'],charge) and np.array_equal(f['spin'],spin) and np.array_equal(f['q0'],q0)
 out=D/'models'/name;out.mkdir(parents=True,exist_ok=True);link=out/'model.npz'
 if link.exists():assert sha(link)==sha(selected)
 else:link.symlink_to(selected)
 selection=dict(molecule=name,original_canonical_gap_hartree=gap,selected_profile='original_qualified_spectral' if gap>0 else 'complete_projected_ground_fock',selected_model=str(selected),selected_model_sha256=sha(selected),response_unchanged=True,reference_excitation_labels_read=False)
 f=D/'selection'/f'{name}.json';f.parent.mkdir(exist_ok=True)
 if f.exists():assert json.loads(f.read_text())==selection
 else:f.write_text(json.dumps(selection,indent=2)+'\n')
 print(json.dumps(selection),flush=True)
