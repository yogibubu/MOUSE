"""Read-only decomposition of completed response-operator corrections."""
from pathlib import Path
import sys,json,ast,importlib.util,socket
import numpy as np
from threadpoolctl import threadpool_limits
from nosi_spectrum import spectrum
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-response-consistent-nosi-20261101');BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-frontier-cisd-20261101');NATIVE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-native-reference-20261101')

def load(name):
 spec=importlib.util.spec_from_file_location('control_'+name,ROOT/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
with threadpool_limits(2):
 p=json.loads((BASE/'protocol.json').read_text());cases=p['cases']+[['prospective_cisd',r['molecule']] for r in p['prospective_cases']]+[['post_kernel_validation',r['molecule']] for r in json.loads((ROOT/'prospective_protocol.json').read_text())['cases']]
 for cohort,name in cases:
    if not (ROOT/cohort/name/'result.json').exists():load('run').run(cohort,name)
    if not (ROOT/'normal_ordered'/cohort/name/'result.json').exists():load('run_normal').run(cohort,name)
    with np.load(ROOT/cohort/name/'checkpoint.npz') as z:delta=z['response_delta'].copy()
    with np.load(ROOT/'normal_ordered'/cohort/name/'checkpoint.npz') as z:delta-=z['projected_one_delta']
    with np.load(BASE/cohort/name/'checkpoint.npz') as z:h=z['matrix'].copy();s=z['metric'].copy()
    old=NATIVE/'nosi_coherent'/cohort/name;oldreport=json.loads((old/'result.json').read_text());rows=oldreport['predictions'];labels=json.loads((BASE/cohort/name/'result.json').read_text())['labels']
    with np.load(old/'checkpoint.npz') as z:reference=np.column_stack([z['coefficients'][:,r['root']] for r in rows])
    with np.load(NATIVE/cohort/name/'coherent_common/ground/stationary.npz') as z:no=int(np.count_nonzero(z['result_occupations'][0]))
    singles=[]
    for i,label in enumerate(labels):
        if label in [r['id'] for r in rows]:singles.append(i)
        elif label.startswith('open_'):
            _,paired,opens=label.split('_',2)
            if 2*(int(paired)>=no)+sum(j>=no for j in ast.literal_eval(opens))==1:singles.append(i)
    controls=[]
    for variant in ['all_transition_components','rank_one_block']:
        dd=delta.copy()
        if variant=='rank_one_block':
            mask=np.zeros_like(dd);mask[np.ix_(singles,singles)]=1;dd*=mask
        diag=np.diag(dd)/np.diag(s);hh=h+dd-.5*s*(diag[:,None]+diag[None,:]);report,e,c=spectrum(hh,s,reference,rows,True)
        controls.append(dict(variant=variant,primary=report));out=ROOT/'kernel_only'/cohort/name;out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/(variant+'.npz'),matrix=hh,metric=s,energies=e,coefficients=c)
        print(name,variant,report['predictions'],flush=True)
    (out/'result.json').write_text(json.dumps(controls,indent=2,default=lambda a:a.tolist())+'\n')
