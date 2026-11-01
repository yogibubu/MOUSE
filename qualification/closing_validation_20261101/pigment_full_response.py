"""Complete native AO-pair CAM response using a density-coordinate chart.

At the anchor B_a=E_a (orthonormal symmetric elementary AO matrices),
A_a=2 Sylvester(S,S)^-1 E_a. This invertible change of the COMPLETE chart
allows quadrature locality without discarding any orbital rotation. Tiny
quadrature AO values are omitted only with explicit directional error checks.
Full charge/spin matrices, linear field, native third order and D4 remain.
"""
import socket,subprocess,argparse,time,json,os
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
assert subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip()=='Mac Studio di Vincenzo'
from pigment_compact_reference import Compact,load_state,HERE
from pigment_reference_embedding import write,sha,power
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.nonlocal_response import FrozenNonlocalResponseReference
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from scipy.linalg import eigh,null_space,solve
from gaussian_embedding_basis import source_molecule
from pyscf import gto,ao2mo
from scipy.optimize import linear_sum_assignment
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from pigment_block_matmul import product

def packed(a,i,j,w):return a[...,i,j]*w

def exchange(case,label,i,j,w,out):
 path=out/(label+'_exchange.npy');progress=out/(label+'_exchange_progress.json');n=case.n;nf=len(i)
 if (out/(label+'_exchange_complete.json')).exists():return np.load(path,mmap_mode='r')
 factors=case.a if label=='coulomb' else case.al;eri_path=out/(label+'_eri.npy')
 if not (out/(label+'_eri_complete.json')).exists():
  factor_path=out/(label+'_native_factors.npy')
  if not factor_path.exists():np.save(factor_path,np.array(factors.reshape(len(factors),n*n),order='C'))
  eri=product(factor_path,factor_path,eri_path);eri.flush();del eri;write(out/(label+'_eri_complete.json'),dict(complete=True))
 eri=np.load(eri_path,mmap_mode='r').reshape(n,n,n,n);matrix=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',shape=(nf,nf),dtype=float);completed=json.loads(progress.read_text())['completed'] if progress.exists() else 0
 for lo in range(completed,nf,128):
  hi=min(nf,lo+128);a=i[lo:hi,None];b=j[lo:hi,None];c=i[None,:];d=j[None,:]
  matrix[lo:hi]=.5*w[lo:hi,None]*w[None,:]*(eri[a,c,b,d]+eri[a,d,b,c]);matrix.flush();write(progress,dict(completed=hi,total=nf))
 write(out/(label+'_exchange_complete.json'),dict(complete=True));return matrix

def calibrate(name,cutoff=1e-12,ri_xc=False):
 root=HERE/'pigment_full_response'/name;out=root/('ri_xc_chart' if ri_xc else 'localized_chart');out.mkdir(parents=True,exist_ok=True)
 if (out/'certificate.json').exists():return
 start=time.monotonic();case=Compact(name);s=case.state;n=case.n;i,j=np.tril_indices(n);w=np.where(i==j,1.,2**.5);nf=len(i)
 # Use the projected raw atomic Gaussian image before global metric
 # reconciliation/orthogonalization. The entire valence span is unchanged.
 chart_path=out/'local_chart.npz'
 if ri_xc:
  R=np.eye(n);inverse_R=np.eye(n)
 elif chart_path.exists():
  with np.load(chart_path) as z:R=z['R'].copy();inverse_R=z['inverse_R'].copy()
 else:
  small,t0=source_molecule(s,raw=True)
  with np.load(HERE/'pigment_dft_reference'/name/'PBE0/reference.npz') as z:cd0=z['coefficients'].copy();sd0=z['overlap'].copy()
  cross=gto.intor_cross('int1e_ovlp',case.mol,small)@t0;cval=cd0[:,case.nc:];raw_image=cval@(cval.T@cross);R=solve(s.overlap,case.lift.T@sd0@raw_image);inverse_R=np.linalg.inv(R);span_delta=case.lift@R-raw_image;assert float(np.max(abs(span_delta.T@sd0@span_delta)))**.5<1e-7
  np.savez(chart_path,R=R,inverse_R=inverse_R)
 for label,attribute in [('coulomb','a'),('long_range','al')]:
  path=out/(label+'_local_factors.npy');source=getattr(case,attribute)
  if not path.exists():
   local=np.lib.format.open_memmap(path,mode='w+',shape=source.shape,dtype=float)
   for lo in range(0,len(source),64):local[lo:lo+64]=R.T@source[lo:lo+64]@R
   local.flush()
  setattr(case,attribute,np.load(path,mmap_mode='r'))
 # Reuse the already completed native Coulomb tensor if available.
 parent_eri=root/'coulomb_eri.npy'
 if not ri_xc and (root/'coulomb_eri_complete.json').exists() and not (out/'coulomb_eri_complete.json').exists():
  original=np.load(parent_eri,mmap_mode='r').reshape((n,)*4);local=ao2mo.incore.full(original,R,compact=False).reshape(n*n,n*n);np.save(out/'coulomb_eri.npy',local);del local;write(out/'coulomb_eri_complete.json',dict(complete=True,reused_native_tensor=True))
 with np.load(case.out/'anchor.npz') as z:embc=z['coefficients'].copy();p=z['density'].copy();ff=z['fock'].copy()
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O',12:'Mg'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ctx=prepare_gfn2_deltascf(calc,s.coordinates_bohr/ANGSTROM_TO_BOHR,orbital_reference=s,allow_large=True)
 state=SimpleNamespace(**vars(s));state.coefficients=embc;kernel=ctx.density_response_kernel(state)
 # Full native response is subtracted at the SAME embedded density as the
 # established calibration; nonlinear native terms still respond afterwards.
 if ri_xc and (root/'coulomb_eri_complete.json').exists():
  for filename in ['coulomb_eri.npy','coulomb_eri_complete.json']:
   if not (out/filename).exists():(out/filename).symlink_to(root/filename)
 native_path=out/'native.npy';progress=out/'native_progress.json';done=out/'native_complete.json'
 if not done.exists():
  matrix=np.load(native_path,mmap_mode='r+') if native_path.exists() else np.lib.format.open_memmap(native_path,mode='w+',shape=(nf,nf),dtype=float);completed=json.loads(progress.read_text())['completed'] if progress.exists() else 0
  for lo in range(completed,nf,16):
   hi=min(nf,lo+16);d=np.zeros((hi-lo,n,n));ids=np.arange(hi-lo);d[ids,i[lo:hi],j[lo:hi]]=1/w[lo:hi];d[ids,j[lo:hi],i[lo:hi]]=1/w[lo:hi];d=R@d@R.T
   response=kernel.response_batch(d) if hasattr(kernel,'response_batch') else np.array([kernel(x) for x in d]);matrix[lo:hi]=packed(R.T@response@R,i,j,w);matrix.flush();write(progress,dict(completed=hi,total=nf))
   if lo%512==0:print(name,'native response',hi,'/',nf,flush=True)
  write(done,dict(complete=True))
 native=np.load(native_path,mmap_mode='r');ks=exchange(case,'coulomb',i,j,w,out);kl=exchange(case,'long_range',i,j,w,out)
 # Quadrature chunks are individually journalled. An interruption cannot
 # add one contribution twice to a partially assembled response matrix.
 ri_report=None
 if ri_xc:
  from pigment_ri_xc import prepare as prepare_ri_xc
  ri_kernels,ri_report=prepare_ri_xc(case,p,i,j,w,out,require_accuracy=False);quad_error=ri_report['maximum_curvature_error_hartree'];chunks=[]
 else:
  rng=np.random.default_rng(71);directions=rng.normal(size=(3,n,n));directions=.5*(directions+directions.transpose(0,2,1));directions/=np.linalg.norm(directions,axis=(1,2))[:,None,None];qx=packed(directions,i,j,w);p=inverse_R@p@inverse_R.T
  xcout=out/'xc';xcout.mkdir(exist_ok=True);direct_total=np.zeros((2,3));projected_total=np.zeros((2,3));chunks=[]
  for block,path in enumerate(case.grid):
   with np.load(path) as z:phi=z['phi']@R;core=z['core'];weight=z['weight']
   rho=.5*(core+case.rho(phi,p));fxc=case.mf._numint.eval_xc_eff('CAM-B3LYP',np.array([rho,rho]),deriv=2,xctype='GGA')[2];fc=.25*(fxc[0,:,0]+fxc[0,:,1]+fxc[1,:,0]+fxc[1,:,1]);fs=.25*(fxc[0,:,0]-fxc[0,:,1]-fxc[1,:,0]+fxc[1,:,1])
   for lo in range(0,len(weight),512):
    hi=min(len(weight),lo+512);tag=f'{block:04d}_{lo:06d}';meta=xcout/(tag+'.json');chunks.append(tag)
    if meta.exists():
     previous=json.loads(meta.read_text());assert previous['cutoff']==cutoff;direct_total+=previous['direct'];projected_total+=previous['projected'];continue
    ph=phi[:,lo:hi];ww=weight[lo:hi];active=np.flatnonzero(np.max(abs(ph),axis=(0,1))>cutoff);ai,aj=np.tril_indices(len(active));pi=active[ai];pj=active[aj];ids=pi*(pi+1)//2+pj;pw=np.where(pi==pj,1.,2**.5);feature=np.empty((4,len(ids),hi-lo));feature[0]=pw[:,None]*ph[0,:,pi]*ph[0,:,pj]
    for k in range(1,4):feature[k]=pw[:,None]*(ph[k,:,pi]*ph[0,:,pj]+ph[0,:,pi]*ph[k,:,pj])
    flat=feature.transpose(1,0,2).reshape(len(ids),-1);drho=np.array([case.rho(ph,dp) for dp in directions]);drho_projected=np.einsum('ia,kag->ikg',qx[:,ids],feature,optimize=True);direct=np.empty((2,3));projected=np.empty((2,3))
    for channel,fx in enumerate([fc[:,:,lo:hi],fs[:,:,lo:hi]]):
     weighted=(np.einsum('ijg,jag->aig',fx,feature,optimize=True)*ww[None,None,:]).reshape(len(ids),-1);matrix=flat@weighted.T;np.save(xcout/(tag+f'_{channel}.npy'),matrix)
     direct[channel]=np.einsum('aig,ijg,ajg,g->a',drho,fx,drho,ww,optimize=True);projected[channel]=np.einsum('aig,ijg,ajg,g->a',drho_projected,fx,drho_projected,ww,optimize=True)
    np.save(xcout/(tag+'_ids.npy'),ids);write(meta,dict(cutoff=cutoff,active_AOs=len(active),pair_features=len(ids),direct=direct.tolist(),projected=projected.tolist()));direct_total+=direct;projected_total+=projected
   print(name,'XC response quadrature',block+1,'/',len(case.grid),flush=True)
  quad_error=float(np.max(abs(direct_total-projected_total)));assert quad_error<1e-7,('Quadrature locality error',quad_error)
 lp_path=out/'packed_coulomb_factors.npy'
 if not lp_path.exists():np.save(lp_path,np.array(case.a[:,i,j]*w[None,:],order='C'))
 raw_j=product(lp_path,lp_path,out/'coulomb_pair.npy');charge=np.lib.format.open_memmap(out/'charge.npy',mode='w+',shape=(nf,nf),dtype=float);charge[:]=raw_j;spin=np.lib.format.open_memmap(out/'spin.npy',mode='w+',shape=(nf,nf),dtype=float)
 for lo in range(0,nf,128):
  hi=min(nf,lo+128);spin[lo:hi]=-.0855*ks[lo:hi]-.207*kl[lo:hi];charge[lo:hi]+=spin[lo:hi]-.5*(native[lo:hi]+native[:,lo:hi].T)
 if ri_xc:
  charge[:]+=ri_kernels[0];spin[:]+=ri_kernels[1]
 for tag in chunks:
  ids=np.load(xcout/(tag+'_ids.npy'));charge[np.ix_(ids,ids)]+=np.load(xcout/(tag+'_0.npy'));spin[np.ix_(ids,ids)]+=np.load(xcout/(tag+'_1.npy'))
 charge.flush();spin.flush()
 # Native-orbital-preserving F/Q levels, the original qualified-pole
 # complement correction, and the already frozen unresolved-block rule.
 with np.load(HERE/'pigment_dft_reference'/name/'CAM/reference.npz') as z:cd=z['coefficients'].copy();ed=z['orbital_energies'].copy();od=z['occupations'].copy();sd=z['overlap'].copy()
 c=s.coefficients;no=s.electron_count//2;nc=np.count_nonzero(od==2)-no;projection=cd.T@sd@case.lift@c;ef=np.diag(c.T@ff@c).copy();eq=ef.copy();mapping=np.zeros(n,int);score=np.zeros(n);margin=np.zeros(n)
 for inds,refs in [(np.arange(no),np.arange(nc,nc+no)),(np.arange(no,n),np.arange(nc+no,len(ed)))]:
  ov=abs(projection[np.ix_(refs,inds)]).T;ii,jj=linear_sum_assignment(-ov**2)
  for a,b in zip(ii,jj):mapping[inds[a]]=refs[b];score[inds[a]]=ov[a,b];margin[inds[a]]=ov[a,b]-np.max(np.delete(ov[a],b),initial=0.)
 good=(score>=.7)&(margin>=.05);gauge=np.mean(ef[:no])-np.mean(ed[mapping[:no]]);eq[good]=ed[mapping[good]]+gauge
 fallback=bool(eq[:no].max()>=eq[no:].min())
 if fallback:eq=ef.copy()
 complement=null_space(projection.T);hpp=projection.T@(ed[:,None]*projection);coupling=projection.T@(ed[:,None]*complement);poles,u=eigh(complement.T@(ed[:,None]*complement));coupling=coupling@u;relax=np.zeros(n);blocks=[]
 for a in range(n):
  if good[a]:
   gap=ed[mapping[a]]-poles;keep=abs(gap)>1e-10;assert np.max(abs(coupling[a,~keep]),initial=0.)<1e-8;relax[a]=np.sum(coupling[a,keep]**2/gap[keep])
  else:
   refs=np.arange(nc,nc+no) if a<no else np.arange(nc+no,len(ed));weights=projection[refs,a]**2;order=np.argsort(-weights);assert weights.sum()>=.49;(length,)=np.shape(order);k=int(np.searchsorted(np.cumsum(weights[order]),.49)+1);ids=refs[order[:k]];weights=weights[order[:k]];mu=float(np.average(ed[ids],weights=weights));eq[a]=mu+gauge;relax[a]=mu-hpp[a,a];blocks.append(dict(native_orbital=a,full_MOs=ids.tolist(),weights=weights.tolist(),centroid_hartree=mu))
 levels=.5*(ef+relax+eq);assert levels[:no].max()<levels[no:].min(),('Corrected native reference loses occupied/virtual gap',levels)
 target=(s.overlap@c*levels)@c.T@s.overlap;pg=2*c[:,:no]@c[:,:no].T;fg,_=ctx.fock_and_energy(np.array([.5*pg,.5*pg]));linear=packed(R.T@(target-fg[0])@R,i,j,w);q0=packed(inverse_R@pg@inverse_R.T,i,j,w)
 operators_path=out/'operators.npy'
 if not (out/'operators_complete.json').exists():
  operators=np.lib.format.open_memmap(operators_path,mode='w+',shape=(nf,n,n),dtype=float);sw,su=eigh(s.overlap);chart_rows=inverse_R@su
  for lo in range(0,nf,32):
   hi=min(nf,lo+32);ui=chart_rows[i[lo:hi]];uj=chart_rows[j[lo:hi]];rotated=.5*w[lo:hi,None,None]*(ui[:,:,None]*uj[:,None,:]+uj[:,:,None]*ui[:,None,:]);rotated*=2/(sw[:,None]+sw[None,:]);operators[lo:hi]=su@rotated@su.T
  operators.flush();write(out/'operators_complete.json',dict(complete=True))
 operators=np.load(operators_path,mmap_mode='r');max_error=0.
 for lo in range(0,nf,64):
  hi=min(nf,lo+64);b=.5*(s.overlap@operators[lo:hi]+operators[lo:hi]@s.overlap);expected=np.zeros_like(b);ids=np.arange(hi-lo);expected[ids,i[lo:hi],j[lo:hi]]=1/w[lo:hi];expected[ids,j[lo:hi],i[lo:hi]]=1/w[lo:hi];expected=inverse_R.T@expected@inverse_R;max_error=max(max_error,float(np.max(abs(b-expected))))
 assert max_error<1e-9
 np.savez(out/'field.npz',linear=linear,q0=q0,F_levels=ef,Q_levels=eq,F_relaxation=relax,common_levels=levels,ground_fock=target,mapping=mapping,qualified=good)
 write(out/'reference_mapping.json',dict(blocks=blocks,molecular_Q_fallback=fallback,weights=[.5,.5],lambda_exchange=.9,eta=.6,offset_ev=0))
 write(out/'certificate.json',dict(host=socket.getfqdn(),wall_seconds=time.monotonic()-start,native_orbitals=n,complete_pair_features=nf,localized_pair_chart=not ri_xc,RI_XC=ri_report,local_chart_condition=float(np.linalg.cond(R)),chart_duality_max_error=max_error,quadrature_cutoff=cutoff,quadrature_directional_curvature_error=quad_error,weights=[.5,.5],lambda_exchange=.9,eta=.6,offset_ev=0,DF_auxbasis='def2-universal-jkfit',native_ground_reused=True,native_stationary_checkpoint_sha256=sha(HERE/'pigment_preflight'/name/'stationary_native.npz'),full_orbital_space=True,reference_energy_labels_read=False,status='Complete charge/spin/one-body reference; spin-pure OO optimization still required'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--wait-for-anchor',action='store_true');p.add_argument('--quadrature-cutoff',type=float,default=1e-12);p.add_argument('--ri-xc',action='store_true');args=p.parse_args();out=HERE/'pigment_full_response'/args.name;out.mkdir(parents=True,exist_ok=True);lock=out/'worker.lock';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  while not (HERE/'pigment_compact_reference'/args.name/'certificate.json').exists():
   if not args.wait_for_anchor:raise RuntimeError('Compact reference incomplete')
   time.sleep(20)
  with threadpool_limits(8):calibrate(args.name,args.quadrature_cutoff,args.ri_xc)
 finally:lock.unlink(missing_ok=True)
