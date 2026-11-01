"""Complete active-space singlets, retaining the already frozen double space.

Extra canonical singles expose missing configuration interaction between
valence promotions. Linear dependence is tested solely by the CSF Gram
matrix; no energy or reference label enters membership selection.
"""
from prepare_fresh import *
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result,save_nosi_result,_cross_metric
from matrix_gfn2._nosi_configurations import from_labels,basis
import shutil

def run(cohort,name,parent_variant='pole_residue_screening'):
 parent=(REPAIR/'prepared_models'/cohort/name if parent_variant=='original_prepared' else HERE/parent_variant/cohort/name);out=HERE/(parent_variant+'_singles')/cohort/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():
  oldmanifest=json.loads((out/'model.json').read_text())
  if oldmanifest['schema']=='mouse.nosi.screened_cisd.v1':return
  for file in ['model.json','checkpoint.npz','result.json']:shutil.copy2(out/file,out/(file+'.before_cisd_schema'))
 model=PreparedNOSIModel(parent/'model.json');case=prepare(cohort,name);r=(load_nosi_result(parent/'checkpoint.npz',model) if (parent/'checkpoint.npz').exists() else model.evaluate(case['calc'],case['xyz'],properties=('energy',)));no=model.no;n=len(model.anchor.overlap);current=list(model.anchor_basis);gram=_cross_metric(current,current);inverse=np.linalg.inv(gram);added=[];excluded=[]
 for hole in range(max(0,no-5),no):
  for particle in range(no,min(n,no+5)):
   label=f'single_h{hole}_p{particle}';candidate=basis(model.arrays['auxiliary_frame'],from_labels([label],no));cross=_cross_metric(current,candidate).ravel();norm=float(_cross_metric(candidate,candidate)[0,0]);residual=norm-cross@inverse@cross
   if residual<1e-6:excluded.append(dict(label=label,residual_norm=float(residual)));continue
   added.append(label);current.extend(candidate);gram=np.block([[gram,cross[:,None]],[cross[None,:],np.array([[norm]])]]);inverse=np.linalg.inv(gram)
 arrays=dict(model.arrays);arrays.update(ground=r.ground_coefficients,states=r.state_coefficients,multipliers=r.multipliers,reference_ci=np.vstack([r.ci_coefficients,np.zeros((len(added),len(r.states)))]));np.savez_compressed(out/'arrays.npz',**arrays)
 for filename in ['template.npz','response.npz']:shutil.copy2(parent/filename,out/filename)
 manifest=dict(model.manifest);manifest['schema']='mouse.nosi.screened_cisd.v1';manifest['augmentation']='frozen_reflection_even_doubles_complete_singles';manifest['labels']=model.labels+added;manifest['sha256']={f:sha(out/f) for f in ['template.npz','response.npz','arrays.npz']};manifest['provenance']=dict(manifest['provenance'],added_singles=added,excluded_linear_dependence=excluded,minimum_gram_eigenvalue=float(np.linalg.eigvalsh(gram).min()),parent_model=model.fingerprint);write(out/'model.json',manifest)
 final=PreparedNOSIModel(out/'model.json');result=final.evaluate(case['calc'],case['xyz'],properties=('energy',));save_nosi_result(out/'checkpoint.npz',result);report=dict(cohort=cohort,molecule=name,states=[dict(id=v.state_id,gap_ev=v.excitation_hartree*EV,overlap=v.overlap,margin=v.margin) for v in result.states],model=final.fingerprint,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,reference_excitation_energies_read=False,added_singles=added,excluded_linear_dependence=excluded,minimum_gram_eigenvalue=float(np.linalg.eigvalsh(gram).min()))
 write(out/'result.json',report);print(name,report['states'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');p.add_argument('--parent',default='pole_residue_screening');a=p.parse_args()
 with threadpool_limits(2):run(a.cohort,a.name,a.parent)
