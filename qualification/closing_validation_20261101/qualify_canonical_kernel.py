"""Compare compiled and reference contractions, including the 64th mask bit."""
from run_pigment_cid import *
from matrix_gfn2._nosi_canonical import block
with threadpool_limits(2):
 rng=np.random.default_rng(914);rows=[]
 for n,no in [(8,4),(32,16)]:
  ground=set(range(no));a=n-1;b=no;h=0;j=no-1
  records=[dict(kind='closed',closed=sorted(ground-{h}|{a})),dict(kind='open',closed=sorted(ground-{h}),open=[h,a])]
  records += [dict(kind='four',closed=sorted(ground-{h,j}),open=[h,j,b,a],component=k) for k in range(2)]
  y=np.linalg.qr(rng.normal(size=(n,n)))[0];eri=rng.normal(size=(n,)*4);eri=.5*(eri+eri.transpose(2,3,0,1));eri=.5*(eri+eri.transpose(1,0,3,2));h1=rng.normal(size=(n,n));h1=(h1+h1.T)/2
  old_s,old_h,_=block(configuration_basis(y,records),y,h1,eri,[],list(range(n)),compiled=False)
  new_s,new_h,details=block(None,y,h1,eri,[],list(range(n)),records=records)
  eh=float(np.max(abs(new_h-old_h)));es=float(np.max(abs(new_s-old_s)));assert eh<1e-9 and es<1e-10,(eh,es)
  rows.append(dict(spin_orbitals=2*n,Hamiltonian_error_hartree=eh,metric_error=es,compiled=details['compiled_integer_kernel']))
 write(HERE/'canonical_kernel_qualification.json',dict(host=socket.getfqdn(),rows=rows,fast_math=False,integral_or_configuration_truncation=False));print(rows,flush=True)
