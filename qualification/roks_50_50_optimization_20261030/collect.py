"""Read completed electronic jobs; calculate only molecular diagnostics on Oracle."""
from pathlib import Path
import socket,json,hashlib
import numpy as np
assert socket.getfqdn()=='oracle.sns.it'
BASE=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030');OLD=Path('/home/enzo/MOUSE-roks-independent-20261030')
refs=json.loads((BASE/'references/catalog.json').read_text())
def load(p):return json.loads(p.read_text()) if p.exists() else None
def angle(a,b,c):
    x=a-b;y=c-b;return float(np.degrees(np.arccos(np.clip(x@y/np.linalg.norm(x)/np.linalg.norm(y),-1,1))))
def metrics(name,z,x):
    d=np.linalg.norm(x[:,None]-x[None,:],axis=-1);ids=lambda atom:np.flatnonzero(z==atom)
    if name=='nitroxyl':
        h,n,o=[int(ids(k)[0]) for k in [1,7,8]];return dict(NH=float(d[h,n]),NO=float(d[n,o]),HNO_angle_degrees=angle(x[h],x[n],x[o]))
    if name=='acetaldehyde':
        o=int(ids(8)[0]);c=int(ids(6)[np.argmin(d[o,ids(6)])]);m=int([i for i in ids(6) if i!=c][0]);h=int(ids(1)[np.argmin(d[c,ids(1)])])
        normal=np.cross(x[m]-x[c],x[h]-x[c]);v=x[o]-x[c]
        eta=float(np.degrees(np.arcsin(np.clip(abs(normal@v)/np.linalg.norm(normal)/np.linalg.norm(v),0,1))))
        return {'C=O':float(d[c,o]),'C-C':float(d[c,m]),'C-Hald':float(d[c,h]),'C-C=O':angle(x[m],x[c],x[o]),'C-C-Hald':angle(x[m],x[c],x[h]),'eta':eta}
    n=int(ids(7)[0]);os=ids(8);cs=ids(6);carbonyl=np.array([cs[np.argmin(d[o,cs])] for o in os]);alkene=np.array([c for c in cs if c not in carbonyl]);hs=ids(1)
    nh=int(hs[np.argmin(d[n,hs])]);ch=[i for i in hs if i!=nh]
    cc=[min(d[c,alkene]) for c in carbonyl]
    return {'N-C':float(np.mean(d[n,carbonyl])),'C=O':float(np.mean([d[c,o] for c,o in zip(carbonyl,os)])),
        'C-C':float(np.mean(cc)),'C=C':float(d[alkene[0],alkene[1]]),'N-H':float(d[n,nh]),'C-H':float(np.mean([min(d[h,alkene]) for h in ch])),
        'C-N-C':angle(x[carbonyl[0]],x[n],x[carbonyl[1]]),
        'N-C=O':float(np.mean([angle(x[n],x[c],x[o]) for c,o in zip(carbonyl,os)])),
        'C-C=C':float(np.mean([angle(x[c],x[alkene[np.argmin(d[c,alkene])]],x[alkene[np.argmax(d[c,alkene])]]) for c in carbonyl]))}
rows=[]
for name in ['nitroxyl','acetaldehyde','maleimide']:
    root=BASE/'results'/name;prepared=load(root/'prepared.json');z=np.array([{'H':1,'C':6,'N':7,'O':8}[a] for a in prepared['atoms']])
    record=dict(molecule=name,branches={})
    for branch in ['main','negative_mode_minus','negative_mode_plus']:
        folder=root/branch
        complete=load(folder/'complete.json');failure=load(folder/'failure.json');summary=load(folder/'optimizer_result.json')
        if not any([complete,failure,summary]):continue
        row=dict(complete=complete,failure=failure,optimizer_summary=summary)
        final=folder/'final.npz'
        if final.exists():
            with np.load(final) as d:xyz=d['coordinates_angstrom'];freq=d['frequencies_cm'];g=d['gradient']
            row.update(geometry=metrics(name,z,xyz),frequencies_cm=freq.tolist(),maximum_gradient=float(np.max(abs(g))),certified_minimum=bool(complete and complete.get('passed')))
            table=refs['geometry_database'].get(name)
            if table:row['geometry_deviation_from_reference']={k:row['geometry'][k]-v for k,v in table.items() if isinstance(v,(float,int))}
        record['branches'][branch]=row
    if (root/'terminal_audit.json').exists():
        audit=load(root/'terminal_audit.json');record['terminal']=audit
        record['terminal']['geometry']=metrics(name,z,np.array(audit['coordinates_angstrom']))
    controls=[]
    for profile in ['full_response','qualified_spectral_response']:
        folder=OLD/'results'/profile/name;td=load(folder/'tda.json');cal=load(folder/'calibration.json')
        state=next(r for r in td['rows'] if r['state']=='n_pi')
        controls.append(dict(profile=profile,tda=td,calibration=cal,excitation_ev=state['energy_ev'],qualified=state['qualified']))
    record['vertical_controls']=controls
    record['vertical_roks_50_50_ev']=prepared['original_vertical_excitation_ev']
    record['vertical_tda_arithmetic_control_ev']=sum(c['excitation_ev'] for c in controls)/2
    record['tbe_vertical_ev']=controls[0]['tda']['rows'][0]['tbe_ev']
    record['certified_minima']=sum(bool(r.get('certified_minimum')) for r in record['branches'].values())
    rows.append(record)
report=dict(host=socket.getfqdn(),molecules=rows,all_prospective_minimum_gates_passed=all(r['certified_minima'] for r in rows),
    invalidated_results=['acetaldehyde/main optimizer_result.json frequencies from the obsolete ground-state fallback: superseded by independent analytic final.npz'],
    library_repairs=['final analytic 50/50 Hessian dispatch','TDA gradient finite-difference fallback preserves excited Hamiltonian','energy-only initial CSF anchor restored without weakening accepted-gradient checkpoint checks'])
(BASE/'results/summary.json').write_text(json.dumps(report,indent=2)+'\n')
for r in rows:print(r['molecule'],'minima',r['certified_minima'],'vertical50',r['vertical_roks_50_50_ev'],'tda',r['vertical_tda_arithmetic_control_ev'])
