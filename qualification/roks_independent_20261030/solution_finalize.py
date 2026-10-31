"""Close exposed response/REKS diagnostics with explicit negative conclusions."""
from remote_bootstrap import BASE,HOST
from pathlib import Path
import sys,json,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
R=BASE/'results'

def stats(errors):
    return dict(count=len(errors),mae_ev=float(np.mean(np.abs(errors))) if errors else None,maxabs_ev=max(map(abs,errors),default=None))

stationary=[]
for p in (R/'stationary_response/stationary_raw').glob('*/*/result.json'):
    a=json.loads(p.read_text());bfile=R/'stationary_response/stationary_spectral'/p.relative_to(R/'stationary_response/stationary_raw')
    if not bfile.exists():continue
    b=json.loads(bfile.read_text());name=a['molecule'];state=a['state']
    old=[json.loads((R/folder/name/state/'result.json').read_text()) for folder in ['full_response','qualified_spectral_response']]
    stationary.append(dict(molecule=name,state=state,tbe_ev=a['tbe_ev'],raw=a,spectral=b,
                           mean_ev=.5*(a['excitation_ev']+b['excitation_ev']),old_mean_ev=.5*sum(x['excitation_ev'] for x in old)))
ss={label:stats([r[label]-r['tbe_ev'] for r in stationary]) for label in ['mean_ev','old_mean_ev']}
write(R/'stationary_response/summary.json',dict(host=HOST,rows=stationary,statistics=ss,
       independent_response_audits_passed=all(json.loads((R/'stationary_response/stationary_raw'/n/'independent_audit.json').read_text())['passed'] for n in ['nitroxyl','uracil']),
       verdict='rejected as universal correction; coherent stationary-density calibration worsens the fixed mean on the exposed six-state screen'))

all_sa=[]
for folder in ['sa_reks_response','sa_reks_response_index1']:
    for p in sorted((R/folder).rglob('result.json')):
        d=json.loads(p.read_text());d['artifact']=str(p.relative_to(BASE));all_sa.append(d)
# Selection depends only on converged state identity and index, never TBE error.
selected=[]
keys=sorted({(d['molecule'],d['state']) for d in all_sa if d['profile']=='qualified_spectral_response'})
for name,state in keys:
    valid=[d for d in all_sa if d['profile']=='qualified_spectral_response' and d['molecule']==name and d['state']==state and d.get('qualified')]
    if valid:
        d=sorted(valid,key=lambda x:(x['observed_index'],x['artifact']))[0]
        old=json.loads((R/'qualified_spectral_response'/name/state/'result.json').read_text())
        raw=json.loads((R/'full_response'/name/state/'result.json').read_text())
        selected.append(dict(molecule=name,state=state,new=d,old_spectral_ev=old['excitation_ev'],old_mean_ev=.5*(raw['excitation_ev']+old['excitation_ev'])))
groups={}
for group,names in [('all_qualified',None),('carbonyl_B',{'acetaldehyde','acetone','formamide'}),('C_exposed',{'glyoxal','pyrazine','tetrazine'}),('D_exposed',{'acrolein','pyrimidine','nitrosomethane'})]:
    rr=[d for d in selected if names is None or d['molecule'] in names]
    groups[group]={label:stats([(d['new']['excitation_ev'] if label=='sa_reks' else d[label])-d['new']['tbe_ev'] for d in rr]) for label in ['sa_reks','old_spectral_ev','old_mean_ev']}
write(R/'sa_reks_response/summary.json',dict(host=HOST,attempts=all_sa,selected=selected,statistics=groups,
       state_selection='first qualified index, not energy/error ranking; all failures retained',
       validation_status='all molecules already exposed; exploratory diagnosis, not independent validation',
       verdict='reject as universal correction: carbonyl improvement does not transfer to acrolein or uracil',
       electronic_response='analytic orbital Hessian with occupation Schur complement; scalar finite-difference audits retained',nuclear_hessian_implemented=False,
       state_interaction_implemented=False,weights_of_existing_composite_changed=False))

probe=json.loads((R/'reks_probe/summary.json').read_text())
maximum=max(d['ground_lowering_ev'] for d in probe['rows'])
d=json.loads((R/'holdout_D_summary.json').read_text())
lines=['# Response and REKS solution audit — 30 October 2026','',
       '## Decision','',
       'Keep the requested fixed 50/50 complete-response/spectral-response energy mean as an experimental baseline. No fitted energy offset, orbital-gap rescaling or molecule-dependent weight is introduced. None of the new alternatives establishes a universal n→pi* correction. No default is changed.','',
       '## Prospective test D','',
       f"The frozen test qualifies all three molecules. MAE: native {d['statistics']['native']['mae_ev']:.6f} eV; complete {d['statistics']['raw_full']['mae_ev']:.6f}; spectral {d['statistics']['spectral']['mae_ev']:.6f}; fixed mean {d['statistics']['energy_mean_50_50']['mae_ev']:.6f}. The original gate fails because the mean does not beat the spectral component. The optimized-weight diagnostics were not applied.",'',
       '## New physical hypotheses tested','',
       f"1. DFT levels and XC response from the same stationary DFT ground density. Independent PySCF response audits pass (largest matrix error below 8e-15). On six exposed states the fixed-mean MAE changes from {ss['old_mean_ev']['mae_ev']:.6f} to {ss['mean_ev']['mae_ev']:.6f} eV. Reject this as a general accuracy correction.",'',
       f"2. REKS(2,2) fixed-orbital ground correlation, with the published fixed interpolation delta=0.4 and six spin-polarized microstates. Across the tested pairs the maximum ground lowering is {maximum:.8f} eV. This fixed-orbital screen does not support a ground-correlation shift of several tenths of an eV. It is not a full REKS prediction.",'',
       '3. Variational SA-REKS(2,2), sharing orbitals between PPS and OSS and optimizing the active occupations. Analytic electronic response includes the occupation Schur complement. Scalar energy checks verify its first and second orbital derivatives. Higher electronic saddle indices are explored when the minimum loses the tracked state. All failed or changed-state roots are retained.','',
       '| Exposed subset | SA-REKS spectral MAE | Original spectral MAE | Original fixed-mean MAE |','|---|---:|---:|---:|']
for group in ['carbonyl_B','C_exposed','D_exposed']:
    s=groups[group];lines.append(f"| {group} | {s['sa_reks']['mae_ev']:.6f} | {s['old_spectral_ev']['mae_ev']:.6f} | {s['old_mean_ev']['mae_ev']:.6f} |")
lines+=['','All values are eV. These SA-REKS results are exposed-data diagnostics, not a second blind D test.','',
        'The carbonyl improvement fails to transfer. Acrolein has a qualified index-1 SA-REKS gap of about 4.171 eV against 3.723 eV. The qualified index-2 uracil n→pi* root is about 5.527 eV against 4.935 eV. The low-energy uracil minimum and several other apparent roots lose CSF identity and cannot count as predictions. SA-REKS is therefore not promoted.','',
        '## Consequences for development','',
        'The evidence does not support solving the residual error by changing the mean, a universal scalar lambda, stationary-density kernel replacement, or the tested two-orbital ensemble alone. A further candidate must change the density/orbital-dependent functional consistently and pass a new frozen independent test; the present data cannot establish which new functional will succeed. Keep ground and excited energies from the same functional and retain explicit spin adaptation and state tracking.','',
        'Existing native and frozen shell/local-tensor ROKS analytic nuclear Hessians remain separate from these nonlocal research profiles. The complete response, spectral response, stationary-density and SA-REKS prototypes are vertical-only. The analytic electronic Hessian in the SA-REKS solver is not an analytic nuclear Hessian. No new production nuclear derivative implementation is delivered by this screen.','',
        'Sources: [REKS analytic response](https://doi.org/10.1063/1.4994542), [DFTB REKS implementation](https://doi.org/10.1021/acs.jctc.9b00132), [official DFTB+ REKS description](https://dftbplus-recipes.readthedocs.io/en/latest/reks/introduction.html).','',
        'All electronic work ran on oracle.sns.it. Existing references and completed checkpoints were reused. See the JSON summaries and solution_provenance.json for exact records and artifact hashes.']
(BASE/'SOLUTION_AUDIT_20261030.md').write_text('\n'.join(lines)+'\n')
print('STATIONARY',ss,'SA',groups,flush=True)
