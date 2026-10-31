"""Cross-basis transition matching, dedicated xTB Gaussian WFNs versus fixed DFT targets.

Integral evaluation and qualification run on Oracle only. TBE labels are never
read. WFN raw primitives are reconstructed exactly, with a metric certificate.
"""
from pathlib import Path
import socket,os,json,re,hashlib,time,argparse
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np
from scipy.io import FortranFile
from pyscf import gto,scf,lib
lib.num_threads(2)
P=Path(__file__).resolve().parent;OUT=P/'external/stda_xtb/attempt_03';REF=json.loads((P/'inputs/target_references.json').read_text());D=json.loads((P/'inputs/SUMMARY.json').read_text());LMN=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1),(3,0,0),(0,3,0),(0,0,3),(2,1,0),(2,0,1),(1,2,0),(0,2,1),(1,0,2),(0,1,2),(1,1,1)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def wfn(path):
 f=FortranFile(path,'r');method=int(f.read_ints(np.int32)[0]);assert method==1
 nat,nbf,nmo,nprim=map(int,f.read_ints(np.int32));names=[f.read_record('S2')[0].decode().strip() for _ in range(nat)];xyz=np.empty((nat,3));z=[]
 for a in range(nat):
  for k in range(3):xyz[a,k]=f.read_reals(np.float64)[0]
  z.append(int(f.read_ints(np.int32)[0]))
 typ=np.array([f.read_ints(np.int32)[0] for _ in range(nprim)])-1;at=np.array([f.read_ints(np.int32)[0] for _ in range(nprim)])-1;ao=np.array([f.read_ints(np.int32)[0] for _ in range(nprim)])-1
 exp=f.read_reals(np.float64);coeff=f.read_reals(np.float64);occ=f.read_reals(np.float64);eps=f.read_reals(np.float64);c=f.read_reals(np.float64).reshape((nbf,nmo),order='F');f.close()
 assert set(ao)==set(range(nbf)) and len(exp)==nprim and c.shape==(nbf,nmo)
 # One primitive angular shell per WFN primitive; unused Cartesian components have zero coefficient.
 basis={f'{sym}{a}':[] for a,sym in enumerate(names)};local=[]
 for k in range(nprim):
  key=f'{names[at[k]]}{at[k]}';local.append(len(basis[key]));basis[key].append([sum(LMN[typ[k]]),[float(exp[k]),1.]])
 mol=gto.M(atom=[(f'{n}{i}',xyz[i]) for i,n in enumerate(names)],basis=basis,unit='Bohr',cart=True,verbose=0)
 shellstart={};s=0
 for a,n in enumerate(names):
  for j in range(len(basis[f'{n}{a}'])):shellstart[a,j]=s;s+=1
 loc=mol.ao_loc_nr(cart=True);probes=xyz+np.array([.13,.17,.19]);values=mol.eval_gto('GTOval_cart',probes);norm=np.zeros((mol.nao_nr(),nbf))
 for k in range(nprim):
  lmn=LMN[typ[k]];l=sum(lmn);cart=[(x,y,l-x-y) for x in range(l,-1,-1) for y in range(l-x,-1,-1)];shell=next(j for j in range(mol.nbas) if mol.bas_atom(j)==at[k] and mol.bas_angular(j)==l and abs(mol.bas_exp(j)[0]-exp[k])<1e-12)
  idx=loc[shell]+cart.index(lmn)
  polynomial=np.prod(np.array([.13,.17,.19])**np.array(lmn))*np.exp(-exp[k]*sum(np.array([.13,.17,.19])**2));normalization=values[at[k],idx]/polynomial
  if not np.isfinite(normalization) or abs(normalization)<1e-20:raise ValueError('unresolved primitive normalization')
  norm[idx,ao[k]]+=coeff[k]/normalization
 cprim=norm@c;overlap=mol.intor_symmetric('int1e_ovlp');err=float(np.max(abs(cprim.T@overlap@cprim-np.eye(nmo))))
 if err>1e-5:raise ValueError('External WFN metric mismatch: '+str(err))
 return mol,cprim,occ,dict(n_atoms=nat,n_contracted_ao=nbf,n_mo=nmo,n_primitive=nprim,reconstructed_ao_count=mol.nao_nr(),maximum_mo_orthonormality_error=err)
def vectors(path,no,nv):
 txt=path.read_text();dim=int(re.search(r'\$tensor space dimension\s+(\d+)',txt)[1]);assert dim==no*nv
 body=txt.split('$eigenpairs',1)[1].split('$end',1)[0];blocks=list(re.finditer(r'^\s*(\d+)\s+eigenvalue\s*=\s*(\S+)\s*$',body,re.M));out=[]
 for k,m in enumerate(blocks):
  end=blocks[k+1].start() if k+1<len(blocks) else len(body);values=[]
  for line in body[m.end():end].splitlines():
   line=line.rstrip()
   for j in range(0,len(line),20):values.append(float(line[j:j+20].replace('D','E')))
  x=np.array(values).reshape(no,nv);assert abs(np.linalg.norm(x)-1)<1e-4
  out.append((int(m[1]),float(m[2].replace('D','E'))*27.211386245988,x))
 return out
def run(name):
 folder=OUT/name;done=folder/'matching.json'
 if done.exists():return
 start=time.perf_counter();index=REF[name];refpath=Path(index['reference']);mol,ce,oe,certificate=wfn(folder/'wfn.xtb');dmol=scf.chkfile.load_mol(str(refpath/'backend.chk'))
 geometry_error=float(np.max(abs(mol.atom_coords()-dmol.atom_coords())))
 if geometry_error>2e-6:raise ValueError('geometries differ')
 certificate['geometry_maximum_difference_bohr']=geometry_error
 certificate['geometry_tolerance_note']='Identical input XYZ; legacy xtb4stda bohr conversion differs at about 1e-7 relative; threshold 2e-6 bohr.'
 with np.load(refpath/'reference.npz') as d:cd=d['coefficients'];od=d['occupations']
 vf=next(p for p in [refpath/'selected_full.npz',refpath/'backend_tda.npz',refpath/'baseline.npz'] if p.exists())
 with np.load(vf) as d:dfvec=d['amplitudes']
 nod=int(np.count_nonzero(od>1));nvd=len(od)-nod;dfvec=dfvec.reshape(len(dfvec),nod,nvd)
 noe=int(np.count_nonzero(oe>1));nve=len(oe)-noe;cross=cd.T@gto.intor_cross('int1e_ovlp',dmol,mol)@ce;oi=cross[:nod,:noe];ov=cross[nod:,noe:];states=vectors(folder/'ciss_a',noe,nve);project=[]
 for root,energy,x in states:
  q=oi@x@ov.T;pn=float(np.linalg.norm(q));project.append((root,energy,q/max(pn,1e-30),pn))
 requested=[r.get('state','n_pi') for r in D['regression_states']+D['prospective_states'] if r['molecule']==name];rows=[]
 for target in requested:
  root=next(t['reference_root'] for t in index['targets'] if t['id']==target);assert root is not None
  v=dfvec[root-1];v=v/np.linalg.norm(v);scores=np.array([abs(np.sum(v*q)) for _,_,q,_ in project]);order=np.argsort(scores)[::-1];i=int(order[0]);best=project[i];margin=float(scores[i]-(scores[order[1]] if len(scores)>1 else 0));qualified=bool(scores[i]>=.7 and margin>=.05 and best[3]>=.7)
  rows.append(dict(molecule=name,state=target,status='matched' if qualified else 'completed_unassigned',assigned_root=best[0],excitation_ev=best[1] if qualified else None,best_overlap=float(scores[i]),margin=margin,projection_norm=best[3],reference_root=root,all_roots=[dict(root=r,excitation_ev=e,overlap=float(sc),projection_norm=pn) for (r,e,_,pn),sc in zip(project,scores)]))
 write(done,dict(host=socket.getfqdn(),method='dedicated sTDA-xTB',metric_certificate=certificate,matching='full DFT transition overlap in exact cross-AO Gaussian metric; no TBE proximity',wfn_sha256=sha(folder/'wfn.xtb'),vectors_sha256=sha(folder/'ciss_a'),reference_vectors_sha256=sha(vf),seconds=time.perf_counter()-start,rows=rows));print(name,[(r['state'],r['status'],r['best_overlap']) for r in rows],flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');a=ap.parse_args();names=sorted(REF) if a.all else ['acetaldehyde']
 for name in names:
  try:run(name)
  except Exception as e:
   import traceback
   write(OUT/name/'matching_failure.json',dict(host=socket.getfqdn(),molecule=name,error=str(e),traceback=traceback.format_exc()));print(name,str(e),flush=True)
