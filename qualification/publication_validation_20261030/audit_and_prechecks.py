"""Remote-only analysis of archived data; no electronic jobs or parameter fitting."""
from pathlib import Path
import socket,json
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
P=Path(__file__).resolve().parent;A=Path('/home/enzo/MOUSE-roks-relative-pair-20261030');O=Path('/home/enzo/MOUSE-roks-independent-20261030');D=json.loads((P/'inputs/SUMMARY.json').read_text())
def write(name,d):(P/name).write_text(json.dumps(d,indent=2)+'\n')
rows=[]
for k,r in enumerate(D['regression_states']+D['prospective_states']):
 n=r['molecule'];st=r.get('state','n_pi');v=[]
 for pr in ['full_response','qualified_spectral_response']:
  f=next(p for p in [A/'results'/pr/n/st/'result.json',O/'results'/pr/n/st/'result.json'] if p.exists());p=json.loads(f.read_text());assert p['qualified'];v.append(p['excitation_ev'])
 assert abs(np.mean(v)-r['excitation_ev'])<1e-7
 rows.append(dict(molecule=n,state=st,full_gap_ev=v[0],spectral_gap_ev=v[1],tbe_ev=r['tbe_ev'],cohort='exposed' if k<15 else 'previous_prospective'))
stats=[]
for w in [0.,.25,.5,.75,1.]:
 err=np.array([w*r['full_gap_ev']+(1-w)*r['spectral_gap_ev']-r['tbe_ev'] for r in rows]);stats.append(dict(weight_full=w,count=len(err),mae_ev=float(np.mean(abs(err))),rmse_ev=float(np.sqrt(np.mean(err**2))),mse_ev=float(np.mean(err)),maxabs_ev=float(max(abs(err)))))
write('weight_anchor_precheck.json',dict(host=socket.getfqdn(),scope='algebraic vertical-anchor energy sensitivity only, not a weighted PES/Hessian',default_changed=False,rows=rows,statistics=stats))
mins={r['molecule']:r for r in D['molecular_minima']}
def xyz(n):
 with np.load(next((A/'molecular'/n).rglob('joint.npz'))) as d:z=d['parent_0_state_atomic_numbers'].astype(int)
 return z,np.array(mins[n]['xyz'])
def ang(a,b,c):
 u=a-b;v=c-b;return float(np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1))))
def oop(v,u,w):return float(np.degrees(np.arcsin(np.clip(abs(v@np.cross(u,w))/np.linalg.norm(v)/np.linalg.norm(np.cross(u,w)),0,1))))
g=[]
for n in ['acetaldehyde','acetone','maleimide']:
 z,x=xyz(n);d=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
 if n!='maleimide':
  o=int(np.flatnonzero(z==8)[0]);c=int(min(np.flatnonzero(z==6),key=lambda i:d[o,i]));cs=[int(i) for i in np.flatnonzero(z==6) if i!=c]
  rs=[('C-O',d[c,o],1.331 if n=='acetaldehyde' else 1.336,'Angstrom'),('C-C',np.mean([d[c,i] for i in cs]),1.508 if n=='acetaldehyde' else 1.507,'Angstrom')]
  if n=='acetaldehyde':
   h=int(min(np.flatnonzero(z==1),key=lambda i:d[c,i]));rs += [('C-Hald',d[c,h],1.091,'Angstrom'),('C-C-O',ang(x[cs[0]],x[c],x[o]),115.2,'degree'),('C-C-Hald',ang(x[cs[0]],x[c],x[h]),119.3,'degree'),('pyramidalization',oop(x[o]-x[c],x[cs[0]]-x[c],x[h]-x[c]),38.7,'degree')]
  else:rs += [('C-C-C',ang(x[cs[0]],x[c],x[cs[1]]),119.5,'degree'),('pyramidalization',oop(x[o]-x[c],x[cs[0]]-x[c],x[cs[1]]-x[c]),42.3,'degree')]
 else:
  ni=int(np.flatnonzero(z==7)[0]);cc=[int(min(np.flatnonzero(z==6),key=lambda i:d[o,i])) for o in np.flatnonzero(z==8)];al=[int(i) for i in np.flatnonzero(z==6) if i not in cc];nh=int(min(np.flatnonzero(z==1),key=lambda i:d[ni,i]));ch=[min(np.flatnonzero(z==1),key=lambda h:d[c,h]) for c in al]
  rs=[('N-C',np.mean([d[ni,c] for c in cc]),1.404,'Angstrom'),('C-O',np.mean([min(d[o,c] for c in cc) for o in np.flatnonzero(z==8)]),1.248,'Angstrom'),('C-C',np.mean([min(d[c,a] for a in al) for c in cc]),1.432,'Angstrom'),('C=C',d[al[0],al[1]],1.386,'Angstrom'),('N-H',d[ni,nh],1.006,'Angstrom'),('C-H',np.mean([d[a,h] for a,h in zip(al,ch)]),1.074,'Angstrom'),('C-N-C',ang(x[cc[0]],x[ni],x[cc[1]]),109.8,'degree')]
 for label,pred,ref,unit in rs:g.append(dict(molecule=n,metric=label,prediction=float(pred),reference=ref,error=float(pred-ref),unit=unit,reference_method='CCSDR(3)/def2-TZVPP',doi='10.1021/acs.jctc.7b00921'))
write('geometry_reference_comparison.json',dict(host=socket.getfqdn(),scope='chemically assigned internal coordinates at three completed minima',rows=g))
f=sorted(mins['nitroxyl']['frequencies_cm']);refs=[990.2,1458.8,3140.5]
write('hno_harmonic_comparison.json',dict(host=socket.getfqdn(),doi='10.1021/acs.jctc.1c00307',reference_method='(T)+EOM/aug-cc-pVTZ',scope='harmonic only; frequency-order correspondence provisional pending mode-vector confirmation; R0 excluded',rows=[dict(label=l,prediction_cm=p,reference_cm=r,error_cm=p-r) for l,p,r in zip(['bend','N-O stretch','N-H stretch'],f,refs)]))
print(json.dumps(dict(statistics=stats,geometry_rows=len(g),electronic_jobs=0)))
