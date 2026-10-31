"""Collect the new functional without hiding failed guards or missing states."""
from remote_bootstrap import BASE,HOST
from pathlib import Path
import json,sys,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest,state_from
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from edft_transition_identity import transition_overlap
R=BASE/'results'

def statistics(values):
    return dict(count=len(values),mae_ev=float(np.mean(np.abs(values))) if values else None,maxabs_ev=max(map(abs,values),default=None),bias_ev=float(np.mean(values)) if values else None)

development=[]
for p in sorted((R/'edft_pbe0_statefollow/qualified_spectral_response').glob('*/n_pi/result.json')):
    d=json.loads(p.read_text());name=d['molecule']
    if name in ['cyclopentadienone','maleimide','pyranone']:continue
    a=json.loads((R/'qualified_spectral_response'/name/'n_pi/result.json').read_text());b=json.loads((R/'full_response'/name/'n_pi/result.json').read_text())
    development.append(dict(molecule=name,candidate=d,old_spectral_ev=a.get('excitation_ev'),old_fixed_mean_ev=.5*(a['excitation_ev']+b['excitation_ev']),tbe_ev=d.get('tbe_ev')))
devstats={label:statistics([(d['candidate']['excitation_ev'] if label=='candidate' else d[label])-d['tbe_ev'] for d in development if d['candidate'].get('qualified')]) for label in ['candidate','old_spectral_ev','old_fixed_mean_ev']}
pi=R/'edft_pbe0_statefollow/qualified_spectral_response/uracil/pi_pi/result.json'
pipi=json.loads(pi.read_text()) if pi.exists() else dict(qualified=False,status='running')
write(R/'edft_pbe0_statefollow/development_summary.json',dict(host=HOST,rows=development,statistics=devstats,pipi_guard=pipi,
       scope='11 previously exposed first-singlet npi cases; exploratory, not independent validation',general_promotion_passed=False,
       no_fitted_energy_offsets=True,existing_composite_weights=[.5,.5],nuclear_derivatives_supported=False))

protocol=BASE/'edft_holdout_F_protocol.json';plan=json.loads(protocol.read_text());held=[];complete=True
for case in plan['cases']:
    name=case['name'];file=R/'holdout_F'/name/'result.json'
    if not file.exists():complete=False;held.append(dict(molecule=name,qualified=False,status='pending'));continue
    d=json.loads(file.read_text());prepared=json.loads((BASE/'prepared_F'/name/'case.json').read_text())
    if not d.get('qualified'):
        held.append(d);continue
    t=prepared['targets'][0];native,xyz=state_from(t);s=native.reference_state
    with np.load(t['transition_file']) as f:anchor=(f['amplitude'],f['occupied'],f['virtual'])
    control_checks={}
    for label in ['raw_full','old_spectral']:
        record=d['models'][label];r=load_derivative_point(record['checkpoint'],request_hash=record['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
        overlap=transition_overlap(r.common_coefficients,t['hole'],t['particle'],s,anchor)
        control_checks[label]=dict(transition_overlap=overlap,passed=bool(overlap>=.7))
    mapping=json.loads((R/'qualified_spectral_response'/name/'mapping.json').read_text())
    mapped=bool(mapping['qualified'][t['hole']] and mapping['qualified'][t['particle']])
    d['control_identity_checks']=control_checks;d['target_orbital_mapping_qualified']=mapped
    d['paired_qualified']=bool(mapped and all(x['passed'] for x in control_checks.values()) and d['models']['candidate'].get('qualified'))
    d['tda']={}
    for label,folder in [('raw_full','full_response'),('old_spectral','qualified_spectral_response')]:
        td=R/folder/name/'tda.json'
        if td.exists():d['tda'][label]=json.loads(td.read_text())
    held.append(d)
paired=[d for d in held if d.get('paired_qualified')];fstats={}
for label in ['native','old_spectral','old_fixed_mean','candidate']:
    errors=[]
    for d in paired:
        value=d['native_roks_ev'] if label=='native' else d['old_fixed_mean_ev'] if label=='old_fixed_mean' else d['models'][label]['excitation_ev']
        errors.append(value-d['tbe_ev'])
    fstats[label]=statistics(errors)
passed=bool(complete and len(paired)==3 and fstats['candidate']['maxabs_ev']<=.5 and all(fstats['candidate']['mae_ev']<fstats[label]['mae_ev'] for label in ['native','old_spectral','old_fixed_mean']))
write(R/'holdout_F_summary.json',dict(host=HOST,complete=complete,planned=3,paired_qualified=len(paired),rows=held,statistics=fstats,passed=passed,
       protocol_sha256=digest(protocol),scope='molecules unused in MOUSE candidate development; independence from published xi training set not certified',
       all_state_promotion=False,nuclear_derivatives_supported=False))
print('EDFT COLLECT',devstats,'F',complete,fstats,'passed',passed,flush=True)
