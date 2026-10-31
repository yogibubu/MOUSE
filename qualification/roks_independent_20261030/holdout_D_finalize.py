"""Report the frozen 50/50 holdout, without changing its acceptance gate."""
from remote_bootstrap import BASE,HOST
from pathlib import Path
import sys,json,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest

R=BASE/'results';names=['acrolein','pyrimidine','nitrosomethane'];rows=[]
for name in names:
    row=json.loads((R/'holdout_D'/name/'result.json').read_text());row['tda']={}
    for label,folder in [('raw_full','full_response'),('spectral','qualified_spectral_response')]:
        td=json.loads((R/folder/name/'tda.json').read_text());row['tda'][label]=td['rows'][0]
    row['tda_arithmetic_control_ev']=.5*sum(t['energy_ev'] for t in row['tda'].values())
    rows.append(row)
stats={}
for label in ['native','raw_full','spectral','energy_mean_50_50','tda_arithmetic_control']:
    errors=[]
    for r in rows:
        if not r['qualified']:continue
        value=(r['native_roks_ev'] if label=='native' else r['ensemble_ev'] if label=='energy_mean_50_50' else
               r['tda_arithmetic_control_ev'] if label=='tda_arithmetic_control' else r['components'][label]['excitation_ev'])
        errors.append(value-r['tbe_ev'])
    stats[label]=dict(count=len(errors),mae_ev=float(np.mean(np.abs(errors))),maxabs_ev=float(max(map(abs,errors))))
all_good=all(r['qualified'] for r in rows)
gate=bool(all_good and stats['energy_mean_50_50']['maxabs_ev']<=.5 and all(stats['energy_mean_50_50']['mae_ev']<stats[k]['mae_ev'] for k in ['native','raw_full','spectral']))
write(R/'holdout_D_summary.json',dict(host=HOST,rows=rows,statistics=stats,passed=gate,protocol_sha256=digest(BASE/'ensemble_holdout_D_protocol.json'),
       fixed_weights=[.5,.5],optimized_weight_diagnostics_applied=False,nuclear_derivatives_supported=False,default_changed=False))
write(R/'ensemble_weight_decision.json',dict(user_instruction='se le differenze sono queste lascia 50/50',weights=[.5,.5],
       fitted_weights_used=False,diagnostics_archived= 'ensemble_holdout_D_weight_protocol.json',reason='explicit user preference; no holdout refitting'))
text=['# Frozen 50/50 independent cohort D', '',
      'Three molecules fixed before their predictions: acrolein, pyrimidine, nitrosomethane. All three state assignments qualify. No fitted energy offsets or post-test weight changes. The previously fitted weight diagnostics remain unused.', '',
      '| Molecule | TBE | Native ROKS | Complete response | Spectral response | Fixed 50/50 |',
      '|---|---:|---:|---:|---:|---:|']
for r in rows:
    text.append(f"| {r['molecule']} | {r['tbe_ev']:.4f} | {r['native_roks_ev']:.4f} | {r['components']['raw_full']['excitation_ev']:.4f} | {r['components']['spectral']['excitation_ev']:.4f} | {r['ensemble_ev']:.4f} |")
text+=['', 'Energies in eV. MAE: '+', '.join(f"{k}={v['mae_ev']:.6f}" for k,v in stats.items())+'.', '',
       '**Original accuracy gate: FAILED.** The 50/50 mean beats native and complete-response ROKS, but does not beat the spectral component. This does not justify adjusting the weight on D.', '',
       'TDA controls use each corrected GFN2 ground-state orbital basis and its own coherent kernel. Their arithmetic mean is a numerical composite control, not TDA of a single variational functional. Preparation costs are distinct from electronic solver timings. These vertical nonlocal models have no implemented nuclear gradients or nuclear Hessian. Existing shell/tensor analytic nuclear Hessians do not qualify these models.', '',
       'See results/holdout_D_summary.json, ensemble_holdout_D_protocol.json and results/ensemble_weight_decision.json for the frozen protocol and full identities.']
(BASE/'HOLDOUT_D_REPORT_20261030.md').write_text('\n'.join(text)+'\n')
print('D FINAL',stats,'passed',gate,flush=True)
