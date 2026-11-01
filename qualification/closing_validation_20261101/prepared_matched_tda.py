"""Matched TDA control on the corrected native-GFN2 reference, not DFT MOs."""
from prepare_fresh import *
from native_batch import install_native_batch
from predict import tda,target_transition
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result

def run(cohort,name,variant):
 root=HERE/variant/cohort/name;output=root/'matched_tda.json'
 if output.exists():return
 model=PreparedNOSIModel(root/'model.json');r=load_nosi_result(root/'checkpoint.npz',model);case=prepare(cohort,name);s=case['s'];no=case['no'];install_native_batch(case,NATIVE/cohort/name);ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=model.anchor,spin_reference=model.response);co=r.ground_coefficients[:,:no];fg,_=ctx.fock_and_energy(np.array([co@co.T]*2));g=SimpleNamespace(fock=fg,coefficients=r.ground_coefficients)
 rows=[r for r in json.loads((CAM/'data'/cohort/'manifest.json').read_text())['states'] if r['molecule']==name];predictions=[]
 for row in rows:
  seed=case['src']/('native_transition_'+row['id']+'.npz')
  if seed.exists():
   with np.load(seed) as z:q=z['amplitude'].copy()
   transition=s.coefficients[:,:no]@q@s.coefficients[:,no:].T
  else:transition=target_transition(case,row)
  start=time.monotonic();value=tda(ctx,g,case['eri'],s,no,transition,exchange_eri=.9*(.19*case['eri']+.46*case['eri_lr']));value.update(id=row['id'],wall_seconds=time.monotonic()-start);predictions.append(value)
 write(output,dict(model=model.fingerprint,predictions=predictions,DFT_spatial_orbitals_used=False,same_corrected_native_reference=True,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,host=socket.getfqdn(),scope='Same corrected native orbitals, one-body F/Q field and charge/spin response. TDA retains its response kernel; the OO-CID screened configuration Hamiltonian is a distinct approximation.'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');p.add_argument('--variant',default='disentangled_bath_eta_0.600000');a=p.parse_args()
 with threadpool_limits(2):run(a.cohort,a.name,a.variant)
