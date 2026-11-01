"""Inspect saved CI vectors and verify the baseline principal submatrix."""
from pathlib import Path
import socket,json,hashlib,argparse
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
import numpy as np
from scipy.linalg import eigh
HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101')

def run(name,frontier,screening_frontier=5):
 parent=HERE/'pigment_cid'/name;folder=parent if frontier==2 else HERE/'pigment_cid_expansion'/name/f'frontier_{frontier}';same_interaction=True
 if screening_frontier!=5:
  folder=HERE/'pigment_cid_expansion'/name/f'screening_{screening_frontier}'/f'frontier_{frontier}'
  parent=HERE/'pigment_cid_expansion'/name/(f'frontier_{frontier}' if frontier==5 else f'screening_{screening_frontier}/frontier_5');same_interaction=frontier!=5
 report=json.loads((folder/'result.json').read_text());baseline=json.loads((parent/'result.json').read_text())
 with np.load(HERE/'pigment_models'/name/'state.npz') as z:no=int(z['electron_count'])//2
 with np.load(folder/'matrices.npz') as z:h=z['h'];s=z['s'];eg=float(z['ground_energy'])
 with np.load(folder/'eigenstates.npz') as z:ci=z['ci']
 count=len(report.get('retained_ROKS_ordinals',[1]));nbase=1+2*count;indices=list(range(1+count,nbase))
 for i,record in enumerate(report['auxiliary_records']):
  rank=2*sum(j>=no for j in record['closed'])+sum(j>=no for j in record.get('open',[]))
  if rank==2:indices.append(nbase+i)
 d=np.eye(len(s))[:,indices];d[0]-=(s[0]@d)/s[0,0];values,u=eigh(d.T@s@d);assert values.min()>1e-10;d=d@((u*values**-.5)@u.T);population=np.sum((d.T@s@ci)**2,axis=0)
 rows=[dict(row,double_subspace_weight=float(weight)) for row,weight in zip(report['states'],population)]
 control=None
 if frontier!=2:
  assert baseline.get('retained_ROKS_ordinals',[1])==report['retained_ROKS_ordinals'];keys=[json.dumps(record,sort_keys=True) for record in report['auxiliary_records']];mapping=list(range(nbase))+[nbase+keys.index(json.dumps(record,sort_keys=True)) for record in baseline['auxiliary_records']]
  with np.load(parent/'matrices.npz') as z:eh=float(np.max(abs(h[np.ix_(mapping,mapping)]-z['h'])));es=float(np.max(abs(s[np.ix_(mapping,mapping)]-z['s'])));ee=abs(eg-float(z['ground_energy']))
  control=dict(Hamiltonian_max_error_hartree=eh,overlap_max_error=es,ground_energy_error_hartree=ee,baseline_CSFs=len(mapping),same_screened_interaction=same_interaction);assert (not same_interaction or eh<1e-8) and es<1e-10 and ee<1e-8,control
  if same_interaction:
   parent_energies=np.array([row['excitation_ev'] for row in baseline['states']]);energies=np.array([row['excitation_ev'] for row in report['states']]);violation=float(np.max(energies[:len(parent_energies)]-parent_energies));control['maximum_Rayleigh_Ritz_violation_ev']=violation;assert violation<1e-7
 analysis=dict(molecule=name,frontier=frontier,screening_frontier=screening_frontier,basis_size=report['basis_size'],states=rows,baseline_principal_submatrix=control,double_population_definition='Metric-orthogonal projector onto retained optimized pairs plus canonical rank-two CSFs; ground projected out',weights=[.5,.5],lambda_exchange=.9,eta=.6,parameters_retuned=False,reference_geometry_changed=False,screened_interaction_changed=not same_interaction,result_sha256=hashlib.sha256((folder/'result.json').read_bytes()).hexdigest())
 (folder/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n');print(json.dumps(dict(frontier=frontier,basis_size=report['basis_size'],control=control,lowest_states=rows[:5]),indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('frontier',type=int);p.add_argument('--screening-frontier',type=int,default=5);a=p.parse_args();run(a.name,a.frontier,a.screening_frontier)
