"""Independent PySCF check of the nonlocal correction to static charge response."""
from pathlib import Path
import socket,os,json,time
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np
from pyscf import gto,scf,dft,lib
from threadpoolctl import threadpool_limits
R=Path('/home/enzo/MOUSE-state-following-20261031');lib.num_threads(2);rows=[]
with threadpool_limits(limits=2):
 for alpha in [1.,.225]:
  mol=gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',basis='sto-3g',verbose=0)
  mf=scf.RHF(mol) if alpha==1 else dft.RKS(mol)
  if alpha!=1:mf.xc='0.225*HF';mf.grids.level=0
  mf.conv_tol=1e-12;mf.kernel();assert mf.converged
  occ=np.flatnonzero(mf.mo_occ==2);virt=np.flatnonzero(mf.mo_occ==0);co=mf.mo_coeff[:,occ];cv=mf.mo_coeff[:,virt];ii=np.repeat(occ,len(virt));aa=np.tile(virt,len(occ));ci=mf.mo_coeff[:,ii];ca=mf.mo_coeff[:,aa];eri=mol.intor('int2e')
  cols=[]
  for i,a in zip(ii,aa):
   v=np.outer(mf.mo_coeff[:,i],mf.mo_coeff[:,a]);dp=.5*(v+v.T);j,k=mf.get_jk(mol,dp,hermi=1);response=j-.5*alpha*k;cols.append(np.einsum('mi,mn,ni->i',ci,response,ca))
  static=np.diag(mf.mo_energy[aa]-mf.mo_energy[ii])+2*np.asarray(cols).T
  direct=np.einsum('mnpq,mi,nj,pa,qb->iajb',eri,co,co,cv,cv,optimize=True).reshape(len(ii),len(ii))
  crossed=np.einsum('mnpq,mi,nb,pj,qa->iajb',eri,co,cv,co,cv,optimize=True).reshape(len(ii),len(ii))
  corrected=static+.5*alpha*(crossed-direct)
  reference_a,reference_b=mf.TDA().get_ab();reference_a=reference_a.reshape(corrected.shape)
  error=float(np.max(abs(corrected-reference_a)));old_error=float(np.max(abs(static-reference_a)))
  assert error<1e-9 and old_error>1e-4,(alpha,error,old_error)
  rows.append(dict(exchange_fraction=alpha,pyscf_A_max_error_hartree=error,uncorrected_static_max_error_hartree=old_error,eigenvalue_max_error_hartree=float(np.max(abs(np.linalg.eigvalsh(corrected)-np.linalg.eigvalsh(reference_a))))))
report=dict(host=socket.getfqdn(),status='passed',cases=rows,scope='independent response-operator validation on water/STO-3G; benchmark TDA still uses corrected GFN2 orbitals')
(R/'tda_exchange_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
