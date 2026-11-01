"""Collective spectral qualification for fragmented native orbital matches.

Keep every individually qualified pole and its frozen complement correction.
For an unresolved orbital, collect the largest full-DFT spectral weights until
its represented norm reaches the same 0.7 amplitude used for a single match.
The normalized spectral block supplies its Q centroid; its Rayleigh relaxation
relative to the original full-DFT image supplies the unresolved F correction.
No excitation energy, bath-efficiency change, or energy offset is an input.
This is a development amendment motivated by the exposed pyrrole extension.
"""
from prepare_fresh import *
from dataclasses import replace
from matrix_gfn2.nosi_surface import PreparedNOSIModel,save_nosi_result
from matrix_gfn2.spin_reference import save_response_reference
import shutil

def run(cohort,name):
 parent=HERE/'downfold_reference'/cohort/name;out=HERE/'spectral_block_reference'/cohort/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 original=PreparedNOSIModel(parent/'model.json');case=prepare(cohort,name);s=case['s'];c=s.coefficients;no=case['no'];n=len(c);cert=json.loads((NATIVE/cohort/name/'coherent_common/reference_certificate.json').read_text());good=np.array(cert['qualified_mapping'],bool);mapping=np.array(cert['mapping'],int)
 with np.load(CAM/cohort/name/'CAM_full_reference/reference.npz') as z:cd=z['coefficients'].copy();ed=z['orbital_energies'].copy();od=z['occupations'].copy()
 nc=np.count_nonzero(od==2)-no;p=cd.T@case['sd']@case['lift']@c;rayleigh=np.sum(ed[:,None]*p*p,axis=0);gauge=np.mean(cert['F_levels'][:no])-np.mean(ed[mapping[:no]]);delta=np.zeros(n);blocks=[]
 for i in np.flatnonzero(~good):
  refs=np.arange(nc,nc+no) if i<no else np.arange(nc+no,len(ed));weights=p[refs,i]**2;order=np.argsort(-weights);total=weights.sum();assert total>=.7**2,(name,i,'insufficient complete occupied/virtual spectral weight',total)
  k=int(np.searchsorted(np.cumsum(weights[order]),.7**2)+1);ids=refs[order[:k]];w=weights[order[:k]];mu=float(np.average(ed[ids],weights=w));q=mu+gauge;df=mu-rayleigh[i];dq=q-cert['Q_levels'][i];delta[i]=.5*(df+dq)
  blocks.append(dict(native_orbital=int(i),full_MOs=ids,weights=w,cumulative_weight=float(w.sum()),centroid_full_hartree=mu,centroid_aligned_hartree=q,spectral_width_hartree=float(np.sqrt(np.average((ed[ids]-mu)**2,weights=w))),F_relaxation_hartree=df,Q_change_hartree=dq))
 dual=np.array([2*solve_sylvester(s.overlap,s.overlap,a) for a in original.response.operators]);dfock=(s.overlap@c*delta)@c.T@s.overlap;dlinear=np.einsum('aij,ij->a',dual,dfock);response=replace(original.response,pair_linear_hartree=original.response.pair_linear_hartree+dlinear,source_model=original.response.source_model+':collective_spectral_blocks')
 save_response_reference(out/'response.npz',response);shutil.copy2(parent/'template.npz',out/'template.npz');arrays=dict(original.arrays);arrays['h']=arrays['h']+(original._anchor_sqrt@c*delta)@c.T@original._anchor_sqrt;np.savez_compressed(out/'arrays.npz',**arrays);manifest=dict(original.manifest);manifest['sha256']={f:sha(out/f) for f in ['template.npz','response.npz','arrays.npz']};manifest['provenance']=dict(manifest['provenance'],parent_model=original.fingerprint,spectral_block_rule='minimal descending spectral-weight set reaching norm 0.7',spectral_blocks=blocks,energy_labels_read=False);write(out/'model.json',manifest)
 final=PreparedNOSIModel(out/'model.json');result=final.evaluate(case['calc'],case['xyz'],properties=('energy',));save_nosi_result(out/'checkpoint.npz',result);report=dict(cohort=cohort,molecule=name,states=[dict(id=s.state_id,gap_ev=s.excitation_hartree*EV,overlap=s.overlap,margin=s.margin) for s in result.states],model=final.fingerprint,delta_common_levels_hartree=delta,spectral_blocks=blocks,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,reference_excitation_energies_read=False,development_amendment=True)
 write(out/'result.json',report);print(name,report['states'],flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('cohort');parser.add_argument('name');a=parser.parse_args()
 with threadpool_limits(2):run(a.cohort,a.name)
