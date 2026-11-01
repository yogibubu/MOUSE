"""Reprepare fixed candidate at displaced anchors and compare identical geometries.

Both in-plane and symmetry-breaking anchors are retained. The prospective
algorithm is unchanged; in particular, missing reflection certification is
reported rather than hand-restoring the larger selected space.
"""
from prepare_fresh import *
from matrix_gfn2.nosi_surface import PreparedNOSIModel

def sensitivity(original_cohort,name,identity):
 out=HERE/'anchors'/name;out.mkdir(parents=True,exist_ok=True)
 original=PreparedNOSIModel(REPAIR/'prepared_models'/original_cohort/name/'model.json');xyz=original.anchor.coordinates_bohr.copy();z=original.anchor.atomic_numbers
 symbols=[{1:'H',6:'C',7:'N',8:'O'}[int(a)] for a in z];carbon=next(i for i,a in enumerate(z) if a==6);oxygen=next(i for i,a in enumerate(z) if a==8)
 normal=original.manifest['provenance']['reflection_certificate'].get('normal');normal=np.linalg.svd(xyz-xyz.mean(0),full_matrices=True)[2][-1] if normal is None else np.array(normal);stretch=xyz[oxygen]-xyz[carbon];stretch/=np.linalg.norm(stretch)
 for direction,vector in [('in_plane_stretch',stretch),('out_of_plane',normal)]:
  for step in [.01,.03]:
   case=f'{name}_{direction}_{step:.2f}_all_states';cohort='closing_anchors';resultpath=out/(case+'.json')
   if resultpath.exists():continue
   displaced=xyz.copy();displaced[carbon]+=step*vector
   xyzfile=out/(case+'.xyz');xyzfile.write_text(str(len(z))+'\nFrozen anchor sensitivity, Angstrom\n'+'\n'.join(f'{s} {r[0]:.12f} {r[1]:.12f} {r[2]:.12f}' for s,r in zip(symbols,displaced/ANGSTROM_TO_BOHR))+'\n')
   for sid in original.state_ids:initial(case,sid,xyzfile,cohort,original.path)
   run(case,identity,xyzfile,cohort,original.path)
   model=PreparedNOSIModel(REPAIR/'prepared_models'/cohort/case/'model.json');calc=GFN2Calculator(symbols,temperature_kelvin=0.)
   # Use precisely the same final coordinates for both Hamiltonian anchors.
   a=original.evaluate(calc,displaced/ANGSTROM_TO_BOHR,properties=('energy',));b=model.evaluate(calc,displaced/ANGSTROM_TO_BOHR,properties=('energy',))
   old=next(s for s in a.states if s.state_id==identity);new=next(s for s in b.states if s.state_id==identity)
   write(resultpath,dict(molecule=name,direction=direction,step_bohr=step,old_anchor_gap_ev=old.excitation_hartree*EV,new_anchor_gap_ev=new.excitation_hartree*EV,anchor_shift_ev=(new.excitation_hartree-old.excitation_hartree)*EV,old_anchor=original.fingerprint,new_anchor=model.fingerprint,old_csf_count=len(original.labels),new_csf_count=len(model.labels),new_reflection=model.manifest['provenance']['reflection_certificate'],state_overlap_original=old.overlap,state_overlap_new=new.overlap,weights=[.5,.5],lambda_exchange=.9,offset_ev=0))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');p.add_argument('identity');a=p.parse_args()
 with threadpool_limits(2):sensitivity(a.cohort,a.name,a.identity)
