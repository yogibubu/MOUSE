"""True singlet TDA A from the real static charge response and bare ERIs."""
from common import *

def complete_tda_A(static,eri,c,no,exchange_fraction=.225):
 co=c[:,:no];cv=c[:,no:];dimension=no*(len(c)-no)
 direct=np.einsum('mnpq,mi,nj,pa,qb->iajb',eri,co,co,cv,cv,optimize=True).reshape(dimension,dimension)
 crossed=np.einsum('mnpq,mi,nb,pj,qa->iajb',eri,co,cv,co,cv,optimize=True).reshape(dimension,dimension)
 correction=.5*exchange_fraction*(crossed-direct);matrix=static+correction
 assert np.max(abs(matrix-matrix.T))<1e-8
 return matrix,correction

def prepared_eri(name,state,model=None):
 if model is not None:return model.arrays['eri']
 p=OLD/f'results/full_response/{name}/projected_eri.npz'
 if p.exists():
  with np.load(p) as a:return a['eri'].copy()
 assert name=='nitroxyl','Missing projected ERI preparation checkpoint'
 dest=OUT/'cost_preparation/nitroxyl/projected_eri.npz'
 if dest.exists():
  with np.load(dest) as a:return a['eri'].copy()
 # Reconstruct only the previously unsaved small ERI tensor. Completed DFT
 # SCF and its existing embedding are reused; no electronic SCF is rerun.
 folder=Path('/home/enzo/MATRIX-runs/tda-dft-independent-20261022-v1/docs/reports/independent_dft_tda_20261022/results/nitroxyl_pbe0')
 embedding=Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel/nitroxyl/kernel.npz')
 with np.load(embedding) as a:lift=a['embedding'].copy();reference_sha=str(a['reference_sha256'])
 assert sha(folder/'reference.npz')==reference_sha
 mol=scf.chkfile.load_mol(str(folder/'backend.chk'));assert np.max(abs(mol.atom_coords()-state.coordinates_bohr))<1e-8
 with np.load(folder/'reference.npz') as a:sd=a['overlap'].copy()
 assert np.max(abs(lift.T@sd@lift-state.overlap))<1e-8
 started=time.monotonic();n=len(state.overlap);eri=ao2mo.kernel(mol,(lift,lift,lift,lift),compact=False).reshape(n,n,n,n)
 dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,eri=eri)
 write(dest.with_suffix('.json'),dict(source='reconstruction of missing projected ERI only',reference_sha256=reference_sha,embedding_sha256=sha(embedding),seconds=time.monotonic()-started,scf_calls=0,source_sha256=sha(__file__)))
 return eri
