"""Fixed BH&HLYP reference survey; no target fitting or geometry changes."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,os,json,numpy as np,time,traceback,argparse
from pyscf import dft,scf,gto,lib
OLD=Path('/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029');sys.path.insert(0,str(OLD))
from frozen_relaxation import plan,state_from,write,digest,key,EV
from extent_support import source_molecule
from matrix_gfn2.hybrid_spectrum import match_orbital_spectrum
XC=os.environ.get('MOUSE_REFERENCE_XC','BHANDHLYP')
if XC not in ['BHANDHLYP','HYB_GGA_XC_LC_WPBE']:raise ValueError('unplanned reference functional')
ROOT=BASE/('results/bhandhlyp' if XC=='BHANDHLYP' else 'results/lc_wpbe');ROOT.mkdir(parents=True,exist_ok=True);lib.num_threads(8)
PROTOCOL=dict(host=HOST,xc=XC,grid_level=3,exact_exchange_ground=(.5 if XC=='BHANDHLYP' else 'range separated, library default omega'),
 lambda_exchange_response=.9,parameters_fitted=0,ground_conv_tol=1e-10,
 ground_conv_tol_grad=3e-8,ground_max_cycle=200,reference_screen='frozen ROKS excitation',
 state_exclusions='preserve previous ambiguous mapping exclusions',
 cases=[c['name'] for c in plan()],no_geometry_optimization=True)

def run(case):
 out=ROOT/case['name'];out.mkdir(exist_ok=True);old=Path(case['reference']);path=out/'reference.npz'
 prior=json.loads((OLD/'frozen_relaxation'/case['name']/'result.json').read_text())
 request=key(dict(protocol=PROTOCOL,reference_sha256=digest(old/'reference.npz'),checkpoint_sha256=digest(old/'backend.chk')))
 mol=scf.chkfile.load_mol(str(old/'backend.chk'));mol.max_memory=20000
 r0,xyz=state_from(next(t for t in case['targets'] if 'checkpoint' in t));s=r0.reference_state
 gs=out/'ground.json';mf=None
 if path.exists() and gs.exists():
  meta=json.loads(gs.read_text());assert meta['request_hash']==request and meta['converged']
  with np.load(path) as f:ref={k:f[k] for k in f.files}
 else:
  with np.load(old/'reference.npz') as f:ref={k:f[k] for k in f.files}
  mf=dft.RKS(mol);mf.xc=XC;mf.verbose=4;mf.max_memory=20000;mf.small_rho_cutoff=0
  mf.grids.level=3;mf.conv_tol=1e-10;mf.conv_tol_grad=3e-8;mf.max_cycle=200;mf.chkfile=str(out/'backend.chk')
  mf._eri=mol.intor('int2e',aosym='s8')
  if (out/'backend.chk').exists():dm=scf.chkfile.load(str(out/'backend.chk'),'scf');dm=(dm['mo_coeff']*dm['mo_occ'])@dm['mo_coeff'].T
  else:dm=(ref['coefficients']*ref['occupations'])@ref['coefficients'].T
  start=time.perf_counter();mf.kernel(dm0=dm)
  if not mf.converged:raise ValueError('DFT ground reference failed to converge')
  ref['coefficients']=mf.mo_coeff;ref['occupations']=mf.mo_occ;ref['orbital_energies_hartree']=mf.mo_energy
  ref['fock']=mf.get_fock();ref['model_id']=np.array(XC+'/aug-cc-pVDZ')
  np.savez_compressed(path,**ref)
  meta=dict(host=HOST,request_hash=request,converged=True,energy_hartree=float(mf.e_tot),
   seconds=time.perf_counter()-start,source_reference_sha256=digest(old/'reference.npz'),reference_sha256=digest(path))
  write(gs,meta)
 u=dft.UKS(mol);u.xc=XC;u.verbose=0;u.max_memory=20000;u.grids.level=3;u.small_rho_cutoff=0
 if mf is not None:u._eri=mf._eri
 else:u._eri=mol.intor('int2e',aosym='s8')
 u.grids.build(with_non0tab=True);h=u.get_hcore();c=ref['coefficients'];occ=ref['occupations'];S=ref['overlap']
 def energy(o):
  dm=np.array([(c*x)@c.T for x in o]);v=u.get_veff(mol,dm)
  return float(u.energy_tot(dm=dm,h1e=h,vhf=v))
 ground=energy(np.array([occ/2]*2));assert abs(ground-meta['energy_hartree'])<1e-7
 small,t=source_molecule(s);cross=(gto.intor_cross('int1e_ovlp',mol,small)@t).T
 mapping=match_orbital_spectrum(s.coefficients,s.overlap,s.alpha_occupations+s.beta_occupations,
  s.orbital_energies_hartree,c,S,occ,ref['orbital_energies_hartree'],cross,reference_model=XC+'/aug-cc-pVDZ')
 rows=[]
 for target in case['targets']:
  row=dict(id=target['id'],tbe_ev=target['tbe_ev']);point=out/(target['id']+'_frozen.json')
  try:
   prev=next(x for x in prior['records'] if x['id']==target['id'])
   if not prev['qualified']:raise ValueError('prior mapping excluded')
   if point.exists():
    row=json.loads(point.read_text());assert row['request_hash']==request
   else:
    hh,pp=target['hole'],target['particle'];mapping.validate_orbitals(np.array([hh,pp]))
    dh,dp=[int(mapping.reference_indices[x]) for x in [hh,pp]]
    om=np.array([occ/2]*2);om[0,dh]=0;om[0,dp]=1;ot=np.array([np.maximum(*om),np.minimum(*om)])
    em,et=energy(om),energy(ot);frozen=(2*em-et-ground)*EV
    hybrid=frozen+prev['se_relaxation_ev']
    row.update(request_hash=request,qualified=True,dft_hole=dh,dft_particle=dp,
     source_hole=hh,source_particle=pp,hole_overlap=float(mapping.overlaps[hh]),particle_overlap=float(mapping.overlaps[pp]),
     ground_hartree=ground,mixed_hartree=em,triplet_hartree=et,frozen_ev=frozen,
     retained_se_relaxation_ev=prev['se_relaxation_ev'],hybrid_ev=hybrid,
     hybrid_error_ev=hybrid-target['tbe_ev'],pbe0_hybrid_ev=prev['hybrid_ev'],
     pbe0_hybrid_error_ev=prev['hybrid_error_ev'])
    write(point,row)
  except Exception as exc:row.update(qualified=False,error=str(exc))
  rows.append(row);write(out/'result.json',dict(host=HOST,protocol=PROTOCOL,ground=meta,records=rows));print('REFERENCE',case['name'],row,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--molecule');a=p.parse_args();write(ROOT/'protocol.json',PROTOCOL)
 for case in plan():
  if a.molecule and case['name']!=a.molecule:continue
  try:run(case)
  except Exception as exc:write(ROOT/case['name']/'failure.json',dict(host=HOST,error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
