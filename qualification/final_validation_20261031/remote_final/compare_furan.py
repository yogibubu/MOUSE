"""Lightweight molecular/symmetry reduction; no electronic computation."""
from pathlib import Path
import json,argparse,numpy as np

def run(root):
 root=Path(root);a=np.load(root/'geometry/Furan/vibrations.npz');x=a['coordinates_angstrom'];modes=a['mass_weighted_modes'];freq=a['frequencies_cm1'];ref=json.loads((Path(__file__).parent/'furan_eom_ccsd_reference.json').read_text())
 assert x.shape==(9,3) and np.max(abs(x[:,0]))<1e-5 and np.max(abs(x[:,1]+x[[1,0,3,2,4,6,5,8,7],1]))<1e-5
 permutation=np.array([1,0,3,2,4,6,5,8,7]);rows=[];groups={k:[] for k in ['a1','a2','b1','b2']}
 for k,f in enumerate(freq):
  v=modes[:,k].reshape(-1,3);plane=float(np.sum(v*v*np.array([-1,1,1])));c2=float(np.sum(v*v[permutation]*np.array([-1,-1,1])))
  assert abs(abs(plane)-1)<1e-5 and abs(abs(c2)-1)<1e-5
  label=('a1' if c2>0 else 'b2') if plane>0 else ('a2' if c2>0 else 'b1');groups[label].append(float(f));rows.append(dict(frequency_cm1=float(f),irrep=label,plane_character=plane,c2_character=c2))
 comparison=[]
 for label,reference in ref['table_VI_cm1'].items():
  values=sorted(groups[label],reverse=True);assert len(values)==len(reference)
  for index,(value,r) in enumerate(zip(values,reference)):comparison.append(dict(irrep=label,within_irrep_frequency_order=index+1,mouse_cm1=value,eom_ccsd_cm1=r,error_cm1=value-r))
 distance=lambda i,j:float(np.linalg.norm(x[i]-x[j]))
 def angle(i,j,k):
  u=x[i]-x[j];v=x[k]-x[j];return float(np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1))))
 bonds={'O-Calpha':np.mean([distance(4,0),distance(4,1)]),'Calpha-Cbeta':np.mean([distance(0,2),distance(1,3)]),'Cbeta-Cbeta':distance(2,3),'Calpha-H':np.mean([distance(0,5),distance(1,6)]),'Cbeta-H':np.mean([distance(2,7),distance(3,8)])}
 angles={'Calpha-O-Calpha':angle(0,4,1),'O-Calpha-Cbeta':np.mean([angle(4,0,2),angle(4,1,3)]),'Calpha-Cbeta-Cbeta':np.mean([angle(0,2,3),angle(1,3,2)]),'O-Calpha-H':np.mean([angle(4,0,5),angle(4,1,6)]),'Calpha-Cbeta-H':np.mean([angle(0,2,7),angle(1,3,8)])}
 br=[dict(coordinate=k,mouse_angstrom=float(v),eom_ccsd_angstrom=ref['table_V_angstrom'][k],error_angstrom=float(v-ref['table_V_angstrom'][k])) for k,v in bonds.items()]
 ar=[dict(coordinate=k,mouse_degrees=float(v),eom_ccsd_degrees=ref['table_V_degrees'][k],error_degrees=float(v-ref['table_V_degrees'][k])) for k,v in angles.items()]
 report=dict(state='Furan C2v A1 valence stationary point',reference_doi=ref['doi'],stationary_point_is_not_minimum=True,nuclear_imaginary_modes=[r for r in rows if r['frequency_cm1']< -20],all_modes=rows,bonds=br,angles=ar,in_plane_frequency_comparison=comparison,bond_mae_angstrom=float(np.mean([abs(r['error_angstrom']) for r in br])),angle_mae_degrees=float(np.mean([abs(r['error_degrees']) for r in ar])),in_plane_frequency_mae_cm1=float(np.mean([abs(r['error_cm1']) for r in comparison])),assignment='exact reflection/C2 characters, then descending frequency within each irrep; no published displacement vectors for cross-method mode-overlap assignment',reference_missing='A2/B1 harmonic values for the C2v A1 state are not published in table VI; no numerical errors assigned to these modes',accuracy_scope='one stationary geometry and fifteen in-plane harmonic values; not full-dimensional minimum/frequency validation')
 (root/'geometry/Furan/reference_comparison.json').write_text(json.dumps(report,indent=2)+'\n');print({k:report[k] for k in ['bond_mae_angstrom','angle_mae_degrees','in_plane_frequency_mae_cm1']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
