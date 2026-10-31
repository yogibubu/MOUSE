"""Oracle candidate: complete projected DFT ground Fock; identical response kernel.
This new one-body profile replaces partial canonical level assignments. It is a
separate exploratory model, not a recovery of the original Q Hamiltonian.
"""
from pathlib import Path
import socket,os,sys,json,hashlib
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MOUSE_VALIDATION_ROOT='/home/enzo/MOUSE-state-following-20261031')
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=R/'ritz_projection_probe';D.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'source/qualification'));import remote_bootstrap
import numpy as np
from scipy.linalg import solve_sylvester
from threadpoolctl import threadpool_limits
from matrix_gfn2.deltascf_derivatives import load_derivative_point
name=sys.argv[1] if len(sys.argv)>1 else 'Bimane_Syn'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
with threadpool_limits(limits=2):
 out=D/'models'/name;out.mkdir(parents=True,exist_ok=True)
 native=P/'blind/prepared'/name/'ground.npz'
 with np.load(native) as f:xyz=f['coordinates_angstrom'].copy();request=str(f['request_hash'])
 s=load_derivative_point(native,request_hash=request,coordinates_angstrom=xyz,require_gradient=False).reference_state
 full=P/'results/full_response'/name
 embedding=full/'embedding.npz'
 if not embedding.exists():embedding=Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/name/'kernel.npz'
 reference=P/'blind/mouse_reference'/name/'reference.npz'
 protocol=dict(schema='mouse.projected_ground_fock.v1',host=socket.getfqdn(),source_sha256=sha(Path(__file__)),full_model_sha256=sha(full/'model.npz'),embedding_sha256=sha(embedding),dft_reference_sha256=sha(reference),native_reference_sha256=sha(native),one_body='L.T @ (S_DFT C_DFT diag(e_DFT) C_DFT.T S_DFT) @ L - F_native; full-spectrum Galerkin projection in the existing isometric embedding',response='exactly unchanged full AO-pair charge and spin response',weights=[.5,.5],lambda_exchange=.9,orbital_gap_scaling=1.,energy_offset_ev=0.,reference_excitation_labels_read=False,exploratory_model_revision=True)
 if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
 else:write(out/'protocol.json',protocol)
 with np.load(full/'model.npz') as f:data={k:f[k].copy() for k in ['operators','charge','spin','q0','linear']}
 with np.load(embedding) as f:lift=f['embedding'].copy()
 with np.load(reference) as f:c=f['coefficients'].copy();sd=f['overlap'].copy();ed=f['orbital_energies'].copy()
 assert np.max(abs(lift.T@sd@lift-s.overlap))<1e-8
 fd=(sd@c*ed)@c.T@sd;fsmall=lift.T@fd@lift;potential=fsmall-s.fock
 dual=np.array([2*solve_sylvester(s.overlap,s.overlap,a) for a in data['operators']]);data['linear']=np.einsum('aij,ij->a',dual,potential)
 np.savez_compressed(out/'model.npz',**data)
 check=s.fock+np.einsum('a,aij->ij',data['linear'],.5*(s.overlap@data['operators']+data['operators']@s.overlap))
 assert np.max(abs(check-fsmall))<1e-8
 write(out/'verification.json',dict(projected_fock_error=float(np.max(abs(check-fsmall))),embedding_metric_error=float(np.max(abs(lift.T@sd@lift-s.overlap))),scalar_offset=0.,full_dft_spectrum_used=True))
 print(json.dumps(protocol,indent=2),flush=True)
