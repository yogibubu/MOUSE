"""Same native GFN2 orbitals and prepared CAM F/Q response; no DFT-MO TDA."""
from prepare_fresh import *
from native_batch import install_native_batch
from predict import tda,target_transition

def run_tda(name,cohort='closing_blind'):
 output=(HERE/'matched_tda' if cohort=='closing_blind' else HERE/'matched_tda'/cohort)/(name+'.json');output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():return
 case=prepare(cohort,name);source=NATIVE/cohort/name;s=case['s'];no=case['no'];install_native_batch(case,source)
 model=source/'coherent_common/model.npz'
 with np.load(model) as z:data={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
 response=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(model));ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=response)
 with np.load(source/'coherent_common/ground/stationary.npz') as z:g=SimpleNamespace(fock=z['result_fock'].copy(),coefficients=z['result_coefficients'].copy())
 rows=[r for r in json.loads((CAM/'data'/cohort/'manifest.json').read_text())['states'] if r['molecule']==name];predictions=[]
 for row in rows:
  seed=case['src']/('native_transition_'+row['id']+'.npz')
  if seed.exists():
   with np.load(seed) as z:q=z['amplitude'].copy()
   transition=s.coefficients[:,:no]@q@s.coefficients[:,no:].T
  else:transition=target_transition(case,row)
  start=time.monotonic();result=tda(ctx,g,case['eri'],s,no,transition,exchange_eri=.9*(.19*case['eri']+.46*case['eri_lr']));result.update(id=row['id'],wall_seconds=time.monotonic()-start);predictions.append(result)
 write(output,dict(predictions=predictions,DFT_spatial_orbitals_used=False,weights=[.5,.5],lambda_exchange=.9,model_sha256=sha(model),host=socket.getfqdn(),scope='same-native-orbital response control, not retuned; OO-CID additionally has screened state interaction'))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--cohort',default='closing_blind');parser.add_argument('--molecule');args=parser.parse_args()
 with threadpool_limits(2):
  for name in ([args.molecule] if args.molecule else ['Formaldehyde','Ethylene','Butadiene']):run_tda(name,args.cohort)
