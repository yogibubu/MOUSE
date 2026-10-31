"""Rebuild extension tables from locked, transferred data; no electronic work."""
from pathlib import Path
import json,shutil,hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parents[1]/'qualification/state_following_20261031/remote/completed_14_state_snapshot'
EXPERIMENT=ROOT.parents[1]/'qualification/pi_pi_repair_20261031/remote_snapshot/pi_pi_repair'
DATA=ROOT/'data/reduced_extension';DATA.mkdir(exist_ok=True)
if SOURCE.exists():
    for name in ['energy_scores.json','energy_scores.csv','prediction_lock.json','labels_opened.json','scope_ledger.json']:
        shutil.copy2(SOURCE/name,DATA/name)
score=json.loads((DATA/'energy_scores.json').read_text());rows=score['rows'];assert len(rows)==14 and all(r['tda_status']=='matched' for r in rows)
families={'n_pi':r'$n\pi^*$', 'pi_pi':r'$\pi\pi^*$', 'CT':'CT','Rydberg':'Rydberg'}
print('families:',sorted({r['family'] for r in rows}))
lines=[r'\begin{table}[!htbp]',r'\centering\small',r'\caption{Completed reduced extension: MAE (eV) on the same geometries and assigned states. F and Q denote the two corrected GFN2 parents; TDA mean is their arithmetic spectral average. This adapted set is separate from the original 19-state cohort.}\label{tab:extension}',r'\begin{tabular}{lrrrrr}',r'\toprule',r'Family & $N$ & OO 50/50 & TDA-F & TDA-Q & TDA mean \\',r'\midrule']
for label,subset in [('All',rows)]+[(families.get(f,f),[r for r in rows if r['family']==f]) for f in dict.fromkeys(r['family'] for r in rows)]:
    maes=[np.mean([abs(r[k]-r['reference_ev']) for r in subset]) for k in ['oo_ev','tda_full_ev','tda_selected_ev','tda_parent_mean_ev']]
    lines.append(f'{label} & {len(subset)} & '+' & '.join(f'{x:.3f}' for x in maes)+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(ROOT/'tables/reduced_extension.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{longtable}{llrrrrr}',r'\caption{Every reduced-extension energy (eV). The original prediction lock precedes label opening. TDA uses the corrected GFN2 ground orbitals.}\label{tab:extensionall}\\',r'\toprule',r'ID / molecule & Family & TBE & OO & TDA-F & TDA-Q & TDA mean \\',r'\midrule\endhead']
for r in rows:
    name=r['molecule'].replace('_',r'\_')
    lines.append(f"{r['state']} {name} & {families.get(r['family'],r['family'])} & "+' & '.join(f'{r[k]:.3f}' for k in ['reference_ev','oo_ev','tda_full_ev','tda_selected_ev','tda_parent_mean_ev'])+r' \\')
lines += [r'\bottomrule',r'\end{longtable}']
(ROOT/'tables/reduced_extension_all.tex').write_text('\n'.join(lines)+'\n')
# A fixed list of converged development attempts, never a minimum over errors.
choices={'B014':['Adenine/0','Adenine/1_retained_v1'],'B016':['Cyclopentadiene/0','Cyclopentadiene/1'],'B022':['Furan/0','Furan/1'],'B024':['Phthalimide/0_fr_v1','Phthalimide/1_gmf_polish_v1'],'B011':['Aza-naphthalene/0','Aza-naphthalene/1']}
expfile=DATA/'multicsf_development.json'
if EXPERIMENT.exists():
    development=[]
    for sid,paths in choices.items():
        rr=next(r for r in rows if r['state']==sid);parents=[]
        for path in paths:
            mol,attempt=path.split('/');p=EXPERIMENT/'optimized'/mol/sid/attempt/'result.json';d=json.loads(p.read_text());assert d['status']=='stationary_candidate';parents.append(d)
        development.append(dict(state=sid,molecule=rr['molecule'],family=rr['family'],reference_ev=rr['reference_ev'],original_oo_ev=rr['oo_ev'],candidate_ev=float(np.mean([p['excitation_ev'] for p in parents])),rank=parents[0]['rank'],parent_results=parents,accuracy_labels_already_exposed=True))
    rr=next(r for r in rows if r['state']=='B015');development.append(dict(state='B015',molecule=rr['molecule'],family=rr['family'],reference_ev=rr['reference_ev'],original_oo_ev=rr['oo_ev'],candidate_ev=rr['oo_ev'],rank=1,parent_results=[],accuracy_labels_already_exposed=True))
    expfile.write_text(json.dumps(dict(scope='95 percent NTO-rank development trial, rejected as universal prescription; Rydberg unresolved attempts archived separately',rows=development),indent=2)+'\n')
exp=json.loads(expfile.read_text())['rows']
lines=[r'\begin{table}[!htbp]',r'\centering\small',r'\caption{Exposed-label minimal-CSF development trial. Off-diagonal bare Slater--Condon couplings and a 95\% NTO-norm rank rule define a new approximate functional. The $n\pi^*$ control invalidates a universal repair claim.}\label{tab:multicsfdev}',r'\begin{tabular}{llrrrr}',r'\toprule',r'Molecule & Family & Rank & TBE & Original OO & Trial OO \\',r'\midrule']
for r in exp:
    lines.append(r['molecule'].replace('_',r'\_')+' & '+families.get(r['family'],r['family'])+f" & {r['rank']} & {r['reference_ev']:.3f} & {r['original_oo_ev']:.3f} & {r['candidate_ev']:.3f}"+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(ROOT/'tables/multicsf_development.tex').write_text('\n'.join(lines)+'\n')
print('pi trial MAE:',np.mean([abs(r['candidate_ev']-r['reference_ev']) for r in exp if r['family']=='pi_pi']))
