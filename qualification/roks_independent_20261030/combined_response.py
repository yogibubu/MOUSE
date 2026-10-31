"""Two parameter-free combinations, frozen before their predictions.

Share identical charge/spin Hessians. Combine one-body terms variationally,
then optimize ground and excited states separately with the same functional.
All nine states are development/exposed data for this new experiment; original
C retains its blind status only for the earlier frozen spectral profile.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json,numpy as np,argparse
from scipy.linalg import solve_sylvester
from threadpoolctl import threadpool_limits
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,digest,key
import full_response as full
from pyscf import lib
lib.num_threads(2)
ROOT=BASE/'results/combined_response';ROOT.mkdir(exist_ok=True)
LABELS=['mixed_functional_50_50','matched_diagonal_full_couplings']
CASES=[c for c in plan() if c['name'] in ['nitroxyl','uracil']]
CASES += [json.loads((BASE/'prepared_B'/n/'case.json').read_text()) for n in ['acetaldehyde','acetone','formamide']]
CASES += [json.loads((BASE/'prepared_C'/n/'case.json').read_text()) for n in ['glyoxal','pyrazine','tetrazine']]
protocol=dict(host=HOST,source_sha256=digest(__file__),scope='all data already exposed; no blind accuracy claim for these combinations',
 profiles=dict(mixed_functional_50_50='arithmetic mean of the two full functionals before orbital optimization, not the mean of stationary excitation energies',
  matched_diagonal_full_couplings='qualified spectral diagonal plus raw full-Fock off-diagonal elements in native canonical orbital basis; unchanged shared charge/spin response'),
 lambda_exchange=.9,gap_scale=1.,state_energy_offset=0.,fitted_parameters=0,state_assignment='same canonical FR-DO/GMF protocol and first qualified root as prior profiles; any known warm stationary branch reported separately, never chosen by TBE error',
 maximum_iterations=100,saddle_indices=[0,1,2,3],minimum_csf_overlap=.7,minimum_reference_transition_overlap=.7,orbital_residual=1e-9,nuclear_derivatives=False,
 models={c['name']:{kind:digest(BASE/'results'/folder/c['name']/'model.npz') for kind,folder in [('raw_full','full_response'),('spectral','qualified_spectral_response')]} for c in CASES})
p=ROOT/'protocol.json'
if p.exists():assert json.loads(p.read_text())==protocol
else:write(p,protocol)

def calibrate(case,s,ctx,label):
 out=ROOT/label/case['name'];out.mkdir(exist_ok=True)
 request=key(dict(protocol=digest(p),models=protocol['models'][case['name']],variant=label))
 if (out/'model.npz').exists():
  with np.load(out/'model.npz') as f:assert str(f['request_hash'])==request;return {k:f[k].copy() for k in f.files if k!='request_hash'},request
 with np.load(BASE/'results/full_response'/case['name']/'model.npz') as f:a={k:f[k].copy() for k in f.files if k!='request_hash'}
 with np.load(BASE/'results/qualified_spectral_response'/case['name']/'model.npz') as f:b={k:f[k].copy() for k in f.files if k!='request_hash'}
 assert all(np.array_equal(a[k],b[k]) for k in ['operators','charge','spin','q0'])
 data=dict(a)
 if label=='mixed_functional_50_50':data['linear']=.5*(a['linear']+b['linear'])
 else:
  basis=.5*(s.overlap@a['operators']+a['operators']@s.overlap)
  va=np.einsum('a,aij->ij',a['linear'],basis);vb=np.einsum('a,aij->ij',b['linear'],basis)
  ha=s.coefficients.T@va@s.coefficients;hb=s.coefficients.T@vb@s.coefficients
  assert np.max(abs(hb-np.diag(np.diag(hb))))<1e-9
  h=ha-np.diag(np.diag(ha))+np.diag(np.diag(hb))
  v=s.overlap@s.coefficients@h@s.coefficients.T@s.overlap
  dual=np.array([2*solve_sylvester(s.overlap,s.overlap,x) for x in data['operators']]);data['linear']=np.einsum('aij,ij->a',dual,v)
  recovered=s.coefficients.T@np.einsum('a,aij->ij',data['linear'],basis)@s.coefficients
  assert np.max(abs(recovered-h))<1e-9
 np.savez_compressed(out/'model.npz',request_hash=request,**data)
 return data,request

def run(label):
 with threadpool_limits(limits=2,user_api='blas'):
  full.ROOT=ROOT/label;full.ROOT.mkdir(exist_ok=True);full.calibrate=lambda case,s,ctx:calibrate(case,s,ctx,label)
  for case in CASES:
   full.run(case)
   for t in case['targets']:
    f=full.ROOT/case['name']/t['id']/'result.json';row=json.loads(f.read_text())
    if row.get('qualified') and t.get('transition_file'):
     from matrix_gfn2.deltascf_derivatives import load_derivative_point
     old,xyz=state_from(t);s=old.reference_state;solved=load_derivative_point(row['checkpoint'],request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
     with np.load(t['transition_file']) as z:amp=z['amplitude'];occ=z['occupied'];virt=z['virtual']
     hole=s.coefficients[:,occ].T@s.overlap@solved.common_coefficients[:,t['hole']];particle=s.coefficients[:,virt].T@s.overlap@solved.common_coefficients[:,t['particle']];rank=np.outer(hole,particle)
     score=float(abs(np.sum(amp*rank))/np.linalg.norm(rank));row['reference_transition_overlap']=score;row['qualified']=bool(score>=.7);write(f,row)
   print('COMBINATION COMPLETED',label,case['name'],flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=LABELS,required=True);args=ap.parse_args();run(args.profile)
