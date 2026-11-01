"""Molecular internal coordinates and mode-character comparisons, remote only.

Experimental fundamentals are explicitly distinguished from harmonic model
frequencies. No empirical frequency scale or energy adjustment is applied.
"""
from pathlib import Path
import json,socket,numpy as np
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101')
def angle(a,b,c):
 u=a-b;v=c-b;return float(np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1))))
def oop(v,u,w):return float(np.degrees(np.arcsin(np.clip(abs(v@np.cross(u,w))/np.linalg.norm(v)/np.linalg.norm(np.cross(u,w)),0,1))))
rows=[];frequency=[];minima=[]
for name in ['Formaldehyde','acetaldehyde']:
 folder=HERE/'structures'/name
 if not (folder/'result.json').exists():continue
 result=json.loads((folder/'result.json').read_text());minima.append(result);x=np.array(result['coordinates_angstrom']);z=np.array(result['atomic_numbers']);d=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2);o=int(np.flatnonzero(z==8)[0]);c=int(min(np.flatnonzero(z==6),key=lambda i:d[o,i]));hs=sorted((int(i) for i in np.flatnonzero(z==1)),key=lambda i:d[c,i]);cs=[int(i) for i in np.flatnonzero(z==6) if i!=c]
 if name=='Formaldehyde':
  refmethod='CC3/aug-cc-pVTZ';refdoi='10.1021/acs.jctc.7b00921';rs=[('C-O',d[c,o],1.326,'Angstrom'),('C-H',np.mean(d[c,hs]),1.089,'Angstrom'),('H-C-H',angle(x[hs[0]],x[c],x[hs[1]]),118.3,'degree'),('pyramidalization',oop(x[o]-x[c],x[hs[0]]-x[c],x[hs[1]]-x[c]),36.8,'degree')]
 else:
  refmethod='CCSDR(3)/def2-TZVPP';refdoi='10.1021/acs.jctc.7b00921';h=hs[0];rs=[('C-O',d[c,o],1.331,'Angstrom'),('C-C',d[c,cs[0]],1.508,'Angstrom'),('C-Hald',d[c,h],1.091,'Angstrom'),('C-C-O',angle(x[cs[0]],x[c],x[o]),115.2,'degree'),('C-C-Hald',angle(x[cs[0]],x[c],x[h]),119.3,'degree'),('pyramidalization',oop(x[o]-x[c],x[cs[0]]-x[c],x[h]-x[c]),38.7,'degree')]
 for label,pred,ref,unit in rs:rows.append(dict(molecule=name,metric=label,prediction=pred,reference=ref,error=pred-ref,unit=unit,reference_method=refmethod,reference_doi=refdoi))
 with np.load(folder/'normal_modes.npz') as a:modes=a['cartesian_modes'].copy();masses=a['masses_amu'].copy();freq=a['frequencies_cm'].copy()
 mw_modes=modes*np.sqrt(np.repeat(masses,3))[:,None]
 def assigned(pattern):
  pattern=pattern.ravel()/np.sqrt(np.repeat(masses,3));pattern/=np.linalg.norm(pattern);score=abs(pattern@mw_modes);i=int(np.argmax(score));return i,float(score[i]**2)
 pattern=np.zeros_like(x);unit=(x[o]-x[c])/d[c,o];pattern[o]=unit;pattern[c]=-unit;i,score=assigned(pattern)
 ref=1173. if name=='Formaldehyde' else 1119.;doi='10.1016/0022-2852(69)90274-4' if name=='Formaldehyde' else '10.1021/ja00402a016'
 frequency.append(dict(molecule=name,mode='C-O stretch',mode_index=i+1,prediction_harmonic_cm=float(freq[i]),reference_fundamental_cm=ref,difference_cm=float(freq[i]-ref),mass_weighted_internal_coordinate_score=score,reference_doi=doi,scope='harmonic versus experimental fundamental; difference includes anharmonicity, not a harmonic error estimate'))
 if name=='Formaldehyde':
  for sign,label,ref in [(1,'symmetric C-H stretch',2847.),(-1,'asymmetric C-H stretch',2968.)]:
   pattern=np.zeros_like(x)
   for h,scale in zip(hs,[1,sign]):
    u=(x[h]-x[c])/d[c,h];pattern[h]=scale*u;pattern[c]-=scale*u
   i,score=assigned(pattern);frequency.append(dict(molecule=name,mode=label,mode_index=i+1,prediction_harmonic_cm=float(freq[i]),reference_fundamental_cm=ref,difference_cm=float(freq[i]-ref),mass_weighted_internal_coordinate_score=score,reference_doi=doi,scope='harmonic versus experimental fundamental; no scale factor'))
report=dict(model_generation='final screened-even OO-CID with unchanged F/Q and lambda',minima=minima,geometries=rows,frequencies=frequency,all_minima_converged=len(minima)==2 and all(r['gradient_max']<2e-5 for r in minima),all_minima_positive=len(minima)==2 and all(r['minimum_frequency_cm']>0 for r in minima),frequency_comparison='Experiment is anharmonic. Inversion/torsion splittings are not treated as harmonic curvature benchmarks. Modes assigned through mass-weighted chemical-coordinate projection, not frequency ordering.')
(HERE/'physical_comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
