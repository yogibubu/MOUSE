"""Select an anchor molecular Q operator by its existing canonical-order gate.
Keep the original qualified spectral operator when it is consistent. Use the
complete projected ground Fock only where the original order check rejects it.
The choice uses ground-reference operators, never excitation labels or errors.
"""
from pathlib import Path
import socket,os,sys,json,hashlib
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MOUSE_VALIDATION_ROOT='/home/enzo/MOUSE-state-following-20261031')
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=R/'consistent_reference';D.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'source/qualification'));import remote_bootstrap
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2.deltascf_derivatives import load_derivative_point
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
protocol=dict(schema='mouse.consistent.reference.v1',host=socket.getfqdn(),source_sha256=sha(Path(__file__)),rule='original qualified spectral operator if its original canonical occupied/virtual gap is strictly positive; otherwise complete projected DFT ground Fock',selection_scope='one operator per molecular frozen anchor, independent of target and excitation labels',weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0.,minimum_gap_hartree=0.,existing_cases_preserved=True,reference_excitation_labels_read=False,exploratory_method_amendment=True)
f=D/'protocol.json'
if f.exists():assert json.loads(f.read_text())==protocol
else:write(f,protocol)
rows=[]
with threadpool_limits(limits=2):
 for name,sid in [('Diazirine','B001'),('Furan','B022'),('Cyclopentadiene','B016'),('Benzoquinone','B039'),('Aza-naphthalene','B011'),('Bimane_Syn','B015')]:
  cp=P/'blind/prepared'/name/'ground.npz'
  with np.load(cp) as f:xyz=f['coordinates_angstrom'].copy();req=str(f['request_hash'])
  s=load_derivative_point(cp,request_hash=req,coordinates_angstrom=xyz,require_gradient=False).reference_state
  original=P/'results/qualified_spectral_response'/name/'model.npz'
  if not original.exists():original=R/'spectral_ordering_probe/models'/name/'model.npz'
  with np.load(original) as f:a=f['operators'];v=f['linear'];old_charge=f['charge'].copy();old_spin=f['spin'].copy();old_q0=f['q0'].copy()
  potential=np.einsum('a,aij->ij',v,.5*(s.overlap@a+a@s.overlap));matrix=s.coefficients.T@(s.fock+potential)@s.coefficients
  assert np.max(abs(matrix-np.diag(np.diag(matrix))))<1e-8
  e=np.diag(matrix);gap=float(np.min(e[s.alpha_occupations==0])-np.max(e[s.alpha_occupations==1]))
  selected=original if gap>0 else R/'ritz_projection_probe/models'/name/'model.npz'
  assert selected.exists(),'build projected Fock fallback first'
  with np.load(selected) as f:
   assert np.array_equal(f['charge'],old_charge) and np.array_equal(f['spin'],old_spin) and np.array_equal(f['q0'],old_q0)
  source=R/('recovery_rank_one' if gap>0 else 'ritz_projection_probe/results')/name/sid/'result.json'
  outcome=json.loads(source.read_text());assert outcome['status']=='matched'
  row=dict(molecule=name,state=sid,original_canonical_gap_hartree=gap,selected_profile='original_qualified_spectral' if gap>0 else 'complete_projected_ground_fock',selected_model=str(selected),selected_model_sha256=sha(selected),prediction_source=str(source),prediction_sha256=sha(source),status=outcome['status'],excitation_ev=outcome['excitation_ev'],joint_residual=outcome['joint_residual'],reused_without_recalculation=True,response_unchanged=True)
  write(D/(name+'.json'),row);rows.append(row)
write(D/'report.json',dict(protocol=protocol,rows=rows,scope='six completed exploratory cases; not six blind accuracy successes',accuracy_scoring_performed=False,limitation='Bimane and aza-naphthalene targeted subsets contain no competing root; global state uniqueness is not established'))
print(json.dumps(rows,indent=2),flush=True)
