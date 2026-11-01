"""Control reference-geometry sensitivity with the defined frozen CSF membership.

Reuse every completed electronic preparation and full canonical matrix. No new
DFT, SCC or stationary optimization is needed to select the original chart.
"""
from prepare_fresh import *
from matrix_gfn2.nosi_surface import PreparedNOSIModel
sys.path.insert(0,str(Q));from nosi_spectrum import spectrum

def fixed(name,original_cohort,identity):
 original=PreparedNOSIModel(REPAIR/'prepared_models'/original_cohort/name/'model.json');directory=HERE/'anchors'/name
 for path in sorted(directory.glob('*all_states.json')):
  case=path.stem;cohort='closing_anchors';manifest=REPAIR/'prepared_models'/cohort/case/'model.json';current=PreparedNOSIModel(manifest);row=json.loads(path.read_text())
  # Correct the reporting-only multi-state index bug using the completed
  # selected eigenproblem. The original implementation/predictions are intact.
  report=json.loads((REPAIR/'even_sector_5'/cohort/case/'result.json').read_text());wanted=next(r for r in report['primary']['predictions'] if r['id']==identity)
  if abs(row['new_anchor_gap_ev']-wanted['gap_ev'])>1e-6:
   archive=HERE/'analysis_before_state_id_fix'/name/path.name;write(archive,row)
   row.update(new_anchor_gap_ev=wanted['gap_ev'],anchor_shift_ev=wanted['gap_ev']-row['old_anchor_gap_ev'],state_id=identity,reporting_correction='select matching state_id instead of states[0]; completed selected-matrix eigenvalue used, no electronic job repeated')
   write(path,row)
  output=HERE/'anchors_fixed_membership'/name/(case+'.json')
  if output.exists():continue
  full=REPAIR/'expanded_cid_fast_5'/cohort/case;parent=json.loads((full/'result.json').read_text())
  from matrix_gfn2.deltascf import prepare_gfn2_deltascf
  symbols=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in current.anchor.atomic_numbers];calc=GFN2Calculator(symbols,temperature_kelvin=0.)
  context=prepare_gfn2_deltascf(calc,current.anchor.coordinates_bohr/ANGSTROM_TO_BOHR,orbital_reference=original.anchor,spin_reference=original.response)
  overlap=current.arrays['auxiliary_frame'].T@context.charge_correction.transport.u@original.arrays['auxiliary_frame'];mapping=np.zeros(len(overlap),int)
  for sector in [np.arange(original.no),np.arange(original.no,len(overlap))]:
   ii,jj=linear_sum_assignment(-abs(overlap[np.ix_(sector,sector)])**2);mapping[sector[jj]]=sector[ii]
  minimum_orbital_overlap=float(np.min(abs(overlap[mapping,np.arange(len(mapping))])))
  import re
  labels=[]
  for label in original.labels:
   match=re.fullmatch(r'pair_h(\d+)_p(\d+)',label)
   if match:label=f'pair_h{mapping[int(match[1])]}_p{mapping[int(match[2])]}'
   else:
    match=re.fullmatch(r'(open|four)_h(\d+)_(\d+)_p(\d+)_(\d+)(?:_([01]))?',label)
    if match:
     hi,hj=sorted([mapping[int(match[2])],mapping[int(match[3])]]);a,b=sorted([mapping[int(match[4])],mapping[int(match[5])]]);label=f'{match[1]}_h{hi}_{hj}_p{a}_{b}'+('' if match[6] is None else '_'+match[6])
   labels.append(label)
  indices=[parent['labels'].index(label) for label in labels]
  with np.load(full/'checkpoint.npz') as z:h=z['matrix'][np.ix_(indices,indices)].copy();metric=z['metric'][np.ix_(indices,indices)].copy()
  old=NATIVE/'nosi_coherent'/cohort/case;rr=json.loads((old/'result.json').read_text())['predictions']
  with np.load(old/'checkpoint.npz') as z:ref=np.column_stack([z['coefficients'][:,r['root']] for r in rr])
  pad=np.zeros((len(indices),len(rr)));pad[:len(ref)]=ref;primary,e,c=spectrum(h,metric,pad,rr,True);ids=list(current.state_ids);byid={r['id']:r for r in primary['predictions']}
  folder=HERE/'fixed_membership_models'/name/case;folder.mkdir(parents=True,exist_ok=True)
  import shutil
  for filename in ['response.npz','template.npz']:
   destination=folder/filename
   if destination.is_symlink():destination.unlink()
   if not destination.exists():shutil.copy2(current.root/filename,destination)
  arrays=dict(current.arrays);arrays['reference_ci']=c[:,[byid[sid]['root'] for sid in ids]];np.savez_compressed(folder/'arrays.npz',**arrays)
  data=dict(current.manifest);data['labels']=labels;data['sha256']={filename:sha(folder/filename) for filename in ['response.npz','template.npz','arrays.npz']};data['provenance']=dict(data['provenance'],membership_frozen_from=original.fingerprint,anchor_selection_changed=False);write(folder/'model.json',data)
  model=PreparedNOSIModel(folder/'model.json');symbols=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in model.anchor.atomic_numbers];calc=GFN2Calculator(symbols,temperature_kelvin=0.);value=model.evaluate(calc,model.anchor.coordinates_bohr/ANGSTROM_TO_BOHR,properties=('energy',));selected=next(s for s in value.states if s.state_id==identity)
  assert abs(selected.excitation_hartree*EV-byid[identity]['gap_ev'])<1e-6
  write(output,dict(molecule=name,state_id=identity,direction=row['direction'],step_bohr=row['step_bohr'],old_anchor_gap_ev=row['old_anchor_gap_ev'],new_anchor_gap_ev=selected.excitation_hartree*EV,anchor_shift_ev=selected.excitation_hartree*EV-row['old_anchor_gap_ev'],frozen_csf_count=len(original.labels),canonical_orbital_mapping=mapping,minimum_orbital_overlap=minimum_orbital_overlap,old_anchor=original.fingerprint,new_anchor=model.fingerprint,model_prepared_without_new_electronic_work=True,selected_predictions=primary['predictions'],scope='same transported CSF membership, re-prepared reference and W; diagnostic of anchor choice within one frozen chart'))
if __name__=='__main__':
 with threadpool_limits(2):
  fixed('Criegee_Intermediate','independent','I002');fixed('Formaldehyde','closing_blind','B001')
