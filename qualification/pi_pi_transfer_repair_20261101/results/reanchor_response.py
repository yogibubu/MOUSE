"""Uniform native-density reanchoring of the prepared DFT second derivative.

Keep the frozen F/Q one-body reference, exchange lambda and all configurations.
This is a fixed-orbital diagnostic until all KKT systems are solved again.
"""
from pathlib import Path
import sys,json,socket,types
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-response-consistent-nosi-20261101');HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-pi-pi-transfer-repair-20261101');CAM=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-cam-response-20261101')
sys.path.insert(0,str(ROOT))
import basis as original
from basis import *
from calibrate import features
from anchor import CompactDFT
from nosi_spectrum import spectrum

def run(item):
 cohort,name=item['cohort'],item['molecule'];out=HERE/'native_reanchor'/cohort/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'fixed_orbital.json').exists():return
 old=original.prepare_basis(cohort,name);case=old['case'];s=case['s'];no=case['no'];c=old['cg'];p=2*c[:,:no]@c[:,:no].T
 model=out/'model.npz'
 with np.load(old['source']/'coherent_common/model.npz') as z:data={k:z[k].copy() for k in z.files}
 if not model.exists():
  functional=CompactDFT(case,CAM/cohort/name/'grid');a=data['operators'];dual=np.array([2*solve_sylvester(s.overlap,s.overlap,x) for x in a]);nf=len(a);n=len(s.overlap);packed=dual.reshape(nf,n*n);eri=case['eri']
  j=packed@eri.reshape(n*n,n*n)@packed.T;k=packed@eri.transpose(0,2,1,3).reshape(n*n,n*n)@packed.T;klr=packed@case['eri_lr'].transpose(0,2,1,3).reshape(n*n,n*n)@packed.T
  charge=j-.0855*k-.207*klr;spin=-.0855*k-.207*klr;completed=0;progress=out/'progress.npz'
  if progress.exists():
   with np.load(progress) as z:charge=z['charge'].copy();spin=z['spin'].copy();completed=int(z['completed'])
  for block,(phi,core,weight) in enumerate(functional.chunks):
   if block<completed:continue
   rho=.5*(core+functional.rho(phi,p));fxc=functional.mf._numint.eval_xc_eff('CAM-B3LYP',np.array([rho,rho]),deriv=2,xctype='GGA')[2]
   fc=.25*(fxc[0,:,0]+fxc[0,:,1]+fxc[1,:,0]+fxc[1,:,1]);fs=.25*(fxc[0,:,0]-fxc[0,:,1]-fxc[1,:,0]+fxc[1,:,1])
   for lo in range(0,len(weight),4096):
    hi=min(lo+4096,len(weight));feat=features(phi[:,lo:hi],dual);flat=feat.transpose(1,0,2).reshape(nf,-1)
    for result,fx in [(charge,fc),(spin,fs)]:
     weighted=(np.einsum('ijg,jag->aig',fx[:,:,lo:hi],feat,optimize=True)*weight[None,None,lo:hi]).reshape(nf,-1);result+=flat@weighted.T
   np.savez_compressed(out/'progress.tmp.npz',charge=charge,spin=spin,completed=block+1);(out/'progress.tmp.npz').replace(progress)
  state=SimpleNamespace(**vars(s));state.coefficients=c;kernel=case['ctx'].density_response_kernel(state);fields=kernel.response_batch(dual);native=np.einsum('aij,bij->ab',dual,fields)
  data['charge']=.5*(charge+charge.T)-native;data['spin']=.5*(spin+spin.T);np.savez_compressed(model,**data)
  write(out/'certificate.json',dict(native_density_anchor=True,fixed_onebody_reference=True,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,new_SCF_calls=0,energy_labels_used=False,old_model_sha256=sha(old['source']/'coherent_common/model.npz'),response_change_norm=float(np.linalg.norm(data['charge']-np.load(old['source']/'coherent_common/model.npz')['charge']))))
 with np.load(model) as z:data={k:z[k].copy() for k in z.files}
 response=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(model));ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=response)
 # Reuse the complete construction and exact CSFs; change only the response provider.
 gl=dict(original.prepare_basis.__globals__);real=gl['prepare_gfn2_deltascf'];gl['prepare_gfn2_deltascf']=lambda *a,**kw:ctx
 import inspect
 text=inspect.getsource(original.prepare_basis).replace('return dict(case=case,','return dict(diagonal=diagonal,four=four,eg=eg,case=case,')
 exec(compile(text,'<native_reanchored_basis>','exec'),gl);d=gl['prepare_basis'](cohort,name)
 with np.load(ROOT/'selected_cid'/cohort/name/'checkpoint.npz') as z:h=z['matrix'].copy();metric=z['metric'].copy();indices=z['indices'].copy()
 diagonal=np.asarray(d['diagonal'])[indices];shift=diagonal-np.diag(h)/np.diag(metric);new=h+.5*metric*(shift[:,None]+shift[None,:]);new[-2:,-2:]=d['four']
 rows=d['rows'];source=d['old']
 with np.load(source/'checkpoint.npz') as z:report=json.loads((source/'result.json').read_text());reference=np.column_stack([z['coefficients'][:,r['root']] for r in report['predictions']])
 pad=np.zeros((len(h),len(rows)));pad[:len(reference)]=reference
 primary,e,v=spectrum(new,metric,pad,rows,True)
 np.savez_compressed(out/'fixed_orbital.npz',matrix=new,metric=metric,energies=e,coefficients=v,indices=indices)
 write(out/'fixed_orbital.json',dict(primary=primary,diagonal_shift_ev=shift*EV,scope='fixed previous orbitals; new stationary solve required'));print(name,primary['predictions'],flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--molecule');args=parser.parse_args()
 with threadpool_limits():
  for item in json.loads((HERE/'cases.json').read_text()):
   if args.molecule is None or item['molecule']==args.molecule:run(item)
