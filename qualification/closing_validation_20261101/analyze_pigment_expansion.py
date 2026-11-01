"""Inspect saved CI vectors and verify the baseline principal submatrix."""
from pathlib import Path
import socket,json,hashlib,argparse,runpy
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
import numpy as np
from scipy.linalg import eigh,cho_factor,cho_solve
from scipy.optimize import linear_sum_assignment
HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101')
runpy.run_path(str(HERE.parents[1]/'source/qualification/remote_bootstrap.py'))
from matrix_gfn2._nosi_selected_roots import projector_population

def run(name,frontier,screening_frontier=5,coherent_parent=False):
 parent=HERE/'pigment_cid'/name;folder=parent if frontier==2 else HERE/'pigment_cid_expansion'/name/f'frontier_{frontier}';same_interaction=True
 if screening_frontier!=5:
  root=HERE/'pigment_cid_expansion'/name
  folder=root/f'screening_{screening_frontier}'/f'frontier_{frontier}'
  candidates=[root/f'screening_{screening_frontier}'/f'frontier_{frontier-1}']
  candidates += [root/f'screening_{w}'/f'frontier_{frontier}' for w in range(screening_frontier-1,5,-1)]
  candidates += [root/f'frontier_{frontier}',HERE/'pigment_cid'/name]
  parent=next(path for path in candidates if (path/'result.json').exists())
  parent_report=json.loads((parent/'result.json').read_text())
  same_interaction=parent_report.get('screening_active_frontier',5)==screening_frontier
 if coherent_parent:
  parent=HERE/f'pigment_cid_expansion/{name}/screening_{frontier-1}/frontier_{frontier-1}';same_interaction=False
 report=json.loads((folder/'result.json').read_text());baseline=json.loads((parent/'result.json').read_text())
 with np.load(HERE/'pigment_models'/name/'state.npz') as z:no=int(z['electron_count'])//2
 with np.load(folder/'matrices.npz') as z:h=z['h'];s=z['s'];eg=float(z['ground_energy'])
 with np.load(folder/'eigenstates.npz') as z:ci=z['ci']
 count=len(report.get('retained_ROKS_ordinals',[1]));nbase=1+2*count;indices=list(range(1+count,nbase))
 for i,record in enumerate(report['auxiliary_records']):
  rank=2*sum(j>=no for j in record['closed'])+sum(j>=no for j in record.get('open',[]))
  if rank==2:indices.append(nbase+i)
 population=projector_population(s,indices,ci)
 rows=[dict(row,double_subspace_weight=float(weight)) for row,weight in zip(report['states'],population)]
 control=None;root_tracking=None
 if frontier!=2:
  assert baseline.get('retained_ROKS_ordinals',[1])==report['retained_ROKS_ordinals'];keys=[json.dumps(record,sort_keys=True) for record in report['auxiliary_records']];mapping=list(range(nbase))+[nbase+keys.index(json.dumps(record,sort_keys=True)) for record in baseline['auxiliary_records']]
  with np.load(parent/'matrices.npz') as z:eh=float(np.max(abs(h[np.ix_(mapping,mapping)]-z['h'])));es=float(np.max(abs(s[np.ix_(mapping,mapping)]-z['s'])));ee=abs(eg-float(z['ground_energy']))
  with np.load(parent/'eigenstates.npz') as z:parent_ci=z['ci'][:,:10]
  cross=ci[:,:10].T@s[:,mapping]@parent_ci;overlap_squared=abs(cross)**2;assert np.max(overlap_squared)<=1+1e-8
  child_ids,parent_ids=linear_sum_assignment(-overlap_squared)
  root_tracking=dict(roots_2_3_principal_cosines_squared=(np.linalg.svd(cross[1:3,1:3],compute_uv=False)**2).tolist(),parent=str(parent.relative_to(HERE)),scope='One-to-one metric-overlap assignment among lowest ten roots',overlap_squared=overlap_squared.tolist(),assignments=[dict(parent_root=int(j+1),child_root=int(i+1),overlap_squared=float(overlap_squared[i,j])) for i,j in zip(child_ids,parent_ids)])
  control=dict(Hamiltonian_max_error_hartree=eh,overlap_max_error=es,ground_energy_error_hartree=ee,baseline_CSFs=len(mapping),same_screened_interaction=same_interaction);assert (not same_interaction or eh<1e-8) and es<1e-10 and ee<1e-8,control
  if same_interaction:
   parent_energies=np.array([row['excitation_ev'] for row in baseline['states']]);energies=np.array([row['excitation_ev'] for row in report['states']]);violation=float(np.max(energies[:min(len(energies),len(parent_energies))]-parent_energies[:min(len(energies),len(parent_energies))]));control['maximum_Rayleigh_Ritz_violation_ev']=violation;assert violation<1e-7
 analysis=dict(molecule=name,frontier=frontier,screening_frontier=screening_frontier,basis_size=report['basis_size'],states=rows,baseline_principal_submatrix=control,root_tracking=root_tracking,double_population_definition='Metric-orthogonal projector onto retained optimized pairs plus canonical rank-two CSFs; ground projected out',weights=[.5,.5],lambda_exchange=.9,eta=.6,parameters_retuned=False,reference_geometry_changed=False,screened_interaction_changed=not same_interaction,result_sha256=hashlib.sha256((folder/'result.json').read_bytes()).hexdigest())
 (folder/('analysis_coherent.json' if coherent_parent else 'analysis.json')).write_text(json.dumps(analysis,indent=2)+'\n');print(json.dumps(dict(frontier=frontier,basis_size=report['basis_size'],control=control,lowest_states=rows[:5]),indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('frontier',type=int);p.add_argument('--coherent-parent',action='store_true');p.add_argument('--screening-frontier',type=int,default=5);a=p.parse_args();run(a.name,a.frontier,a.screening_frontier,a.coherent_parent)
