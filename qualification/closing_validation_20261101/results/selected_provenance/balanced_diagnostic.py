"""Compare full spin-pure CI excitation gaps to the projected/native-ground gaps.

The correlated ground and excited eigenvalues come from exactly the same
Hermitian Hamiltonian and metric. This is an energy-reference consistency
control, not an additive fitted offset. No energy label is read here.
"""
from prepare_fresh import *
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result
from matrix_gfn2._nosi_surface_response import MatrixResponse
from scipy.linalg import eigh

def run_balanced(cohort,name,reference):
 root=HERE/'downfold_reference' if reference=='diagonal_self_energy' else HERE/'downfold_canonical_spectral' if reference=='canonical_spectral' else REPAIR/'prepared_models'
 folder=root/cohort/name;model=PreparedNOSIModel(folder/'model.json');destination=HERE/('balanced_'+reference)/cohort/name;destination.mkdir(parents=True,exist_ok=True)
 if (destination/'result.json').exists():return
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.)
 r=load_nosi_result(folder/'checkpoint.npz',model) if (folder/'checkpoint.npz').exists() else model.evaluate(calc,model.anchor.coordinates_bohr/ANGSTROM_TO_BOHR,properties=('energy',))
 context=prepare_gfn2_deltascf(calc,r.coordinates_angstrom,orbital_reference=model.anchor,spin_reference=model.response);a=model.arrays;response=MatrixResponse(context,model.response,r.ground_coefficients,r.state_coefficients,r.multipliers,a['auxiliary_frame'],model.labels,a['h'],a['eri'],derivatives=False,auxiliary_records=model.auxiliary_records);h,s,eg=response.matrices(np.zeros(0));e,c=eigh(h.v,s.v);overlap=abs(r.ci_coefficients.T@s.v@c);ii,jj=linear_sum_assignment(-overlap[:,1:]);rows=[]
 for i,j in zip(ii,jj+1):
  score=overlap[i,j];margin=score-np.max(np.delete(overlap[i,1:],j-1),initial=0.);rows.append(dict(id=r.states[i].state_id,gap_ev=float((e[j]-e[0])*EV),root=int(j),overlap=float(score),margin=float(margin),qualified=bool(score>=.7 and margin>=.05)))
 ground_character=float(abs(s.v[0]@c[:,0])/np.sqrt(s.v[0,0]));report=dict(molecule=name,cohort=cohort,reference=reference,predictions=rows,ground_character=ground_character,ground_correlation_ev=float((e[0]-r.ground_energy_hartree)*EV),native_ground_energy=r.ground_energy_hartree,ground_energy=float(e[0]),energy_labels_read=False,model_sha256=model.fingerprint,offset_ev=0,weights=[.5,.5],lambda_exchange=.9,full_singlet_metric_min=float(np.linalg.eigvalsh(s.v).min()))
 np.savez_compressed(destination/'checkpoint.npz',matrix=h.v,metric=s.v,energies=e,coefficients=c);write(destination/'result.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');p.add_argument('--reference',choices=['diagonal_self_energy','canonical_spectral','original'],default='diagonal_self_energy');args=p.parse_args()
 with threadpool_limits(2):run_balanced(args.cohort,args.name,args.reference)
