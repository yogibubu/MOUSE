"""Lightweight result inspection only; this script runs no electronic jobs."""
from pathlib import Path
import json,statistics,math,hashlib
ROOT=Path(__file__).resolve().parent
REMOTE=ROOT/'remote'

def read(path):return json.loads(path.read_text())
def statistics_for(rows):
 errors=[r['excitation_ev']-r['tbe_ev'] for r in rows]
 return dict(count=len(errors),mae_ev=sum(map(abs,errors))/len(errors),maximum_absolute_error_ev=max(map(abs,errors)),mean_error_ev=sum(errors)/len(errors))
regression=[read(p) for p in sorted((REMOTE/'vertical').glob('*/*/result.json'))]
prospective=[read(p) for p in sorted((REMOTE/'validation_G').glob('*/result.json'))]
assert len(regression)==15 and all(r['qualified'] for r in regression)
assert len(prospective)==4 and all(r['qualified'] for r in prospective)
assert max(abs(r['independent_vertical_energy_change_hartree']) for r in prospective)<1e-9
molecular=[read(p) for p in sorted((REMOTE/'molecular').glob('*/complete.json'))]
assert len(molecular)==5 and all(r['passed'] for r in molecular)
columns=[read(p) for p in sorted((REMOTE/'molecular').glob('*/full_hessian_columns/result.json'))]
assert len(columns)==2 and all(r['passed'] for r in columns)
timing=[]
for p in sorted((REMOTE/'vertical').glob('*/n_pi/matched_timing_with_preparation.json')):
 r=read(p);timing.append(dict(molecule=r['molecule'],tda_two_all_roots_seconds=statistics.median(x['tda_both_all_roots']['seconds'] for x in r['rows']),roks_one_state_warm_seconds=statistics.median(x['roks_one_state_warm']['seconds'] for x in r['rows']),maximum_tda_matrix_error=max(x['maximum_tda_matrix_error'] for x in r['rows'])))
assert len(timing)==5
transfer=read(ROOT/'TRANSFER_VERIFICATION.json')
assert transfer['all_artifact_hashes_match'] and transfer['all_source_hashes_match'] and transfer['example_restart']['passed']
assert transfer['remote_full_tests'].startswith('86 passed')
allrows=regression+prospective
npi=[r for r in allrows if r.get('state','n_pi')=='n_pi']
tda_stats={}
for profile in ['full_response','qualified_spectral_response']:
 values=[next(t['energy_ev'] for t in r['tda_gfn2'] if t['profile']==profile)-r['tbe_ev'] for r in prospective]
 tda_stats[profile]=dict(count=len(values),mae_ev=sum(map(abs,values))/len(values),maximum_absolute_error_ev=max(map(abs,values)))
summary=dict(schema='mouse.roks.relative_active_pair.validation.v1',compute_host='oracle.sns.it',weights=[.5,.5],lambda_exchange=.9,energy_offset=0.,
 method='fixed squared relative overlaps of the assigned open orbital pairs; unchanged independent energy mean; exact KKT orbital/nuclear response',
 regression_statistics=statistics_for(regression),prospective_statistics=statistics_for(prospective),all_statistics=statistics_for(allrows),npi_statistics=statistics_for(npi),
 prospective_tda_gfn2_statistics=tda_stats,regression_states=regression,prospective_states=prospective,
 molecular_minima=molecular,software_and_transfer_validation=transfer,full_hessian_column_checks=columns,timing_with_preparation=timing,
 limitations=['State-coherent constrained composite PES, not an unconstrained single DFT functional.','Four prospective molecules have prior native/shell predictions; none was used to fit this composite.','Residual error is not zero: this is a validated candidate, not a demonstrated universal correction.','Cached energy-only ROKS is slower than the two-parent TDA control in this implementation.'])
(ROOT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# ROKS 50/50 con coerenza relativa degli stati','',
'La superficie composita conserva esattamente le energie verticali del 50/50 indipendente e mantiene durante il rilassamento gli overlap relativi dei due orbitali aperti. Le due CSF restano singoletti puri; i vincoli sono ricavati dalle funzioni d’onda assegnate, senza utilizzare le energie di riferimento. Il peso resta 1/2, λ=0,9 scala una sola volta lo scambio della risposta e non i gap. Non si aggiunge alcun offset.',
'',
'Il modello è una superficie composita vincolata, non il ROKS non vincolato di un singolo funzionale DFT. Gli orbitali h/p mantengono le etichette della promozione assegnata all’ancora. Gli angoli relativi sono costanti; le orientazioni assolute e tutti gli orbitali spettatori possono rilassare. I casi di overlap unitario sono eliminati analiticamente dal sistema KKT, evitando Jacobiane singolari.',
'',
'## Verifica quantitativa','',
f"- 19 stati / 18 molecole: MAE {summary['all_statistics']['mae_ev']:.4f} eV; 18 n→π*: MAE {summary['npi_statistics']['mae_ev']:.4f} eV.",
f"- Errore massimo residuo: {summary['all_statistics']['maximum_absolute_error_ev']:.4f} eV. L’universalità della correzione non è dimostrata.",
'- Cinque minimi molecolari con tutti i modi reali, gap positivi e controlli indipendenti delle derivate.',
'- 86 test remoti passati; avvio automatico da due checkpoint qualificati, tracking APOC e restart atomico.',
f"- Esempio pubblico verificato sia al primo avvio sia da restart: differenza energetica {transfer['example_restart']['energy_change_hartree']:.2e} Ha; {transfer['verified_files']} file dei risultati trasferiti con checksum corrispondenti.",
'',
'## Quattro molecole prospettiche','',
'Questi sistemi non erano usati per sviluppare i due modelli non locali o il vincolo relativo. Le precedenti predizioni native/shell esistevano già. Il protocollo è stato fissato prima delle nuove predizioni; non si presenta il campione come una banca dati mai esaminata.',
'',
'| Molecola n→π* | Riferimento | ROKS 50/50 | TDA completa su GFN2 | TDA spettrale su GFN2 |',
'|---|---:|---:|---:|---:|']
for r in prospective:
 t={x['profile']:x['energy_ev'] for x in r['tda_gfn2']}
 lines.append(f"| {r['molecule']} | {r['tbe_ev']:.3f} | {r['excitation_ev']:.3f} | {t['full_response']:.3f} | {t['qualified_spectral_response']:.3f} |")
lines += ['',f"MAE sui quattro sistemi: **{summary['prospective_statistics']['mae_ev']:.3f} eV** per ROKS; **{tda_stats['full_response']['mae_ev']:.3f} / {tda_stats['qualified_spectral_response']['mae_ev']:.3f} eV** per le due TDA. Tutti gli stati passano i controlli della transizione DFT proiettata. La TDA usa gli orbitali fondamentali GFN2 corretti del rispettivo modello, con lo stesso kernel; nessun orbitale DFT entra nell’equazione TDA. La media delle due TDA è solo un controllo composito, non la risposta lineare unica del sistema KKT ROKS.",
'', '## Minimi e Hessiane analitiche','',
'| Molecola | Gradiente massimo (Ha/bohr) | Frequenza minima (cm⁻¹) | Errore H·v a 10⁻⁴ bohr (Ha/bohr²) |',
'|---|---:|---:|---:|']
for r in molecular:lines.append(f"| {r['molecule']} | {r['maximum_gradient']:.2e} | {min(r['frequencies_cm']):.2f} | {r['checks'][-1]['hvp_error']:.2e} |")
lines += ['',
'Le Hessiane comprendono la risposta KKT degli orbitali, i moltiplicatori dei vincoli, il rilassamento dei due modelli e tutte le derivate delle tre superfici S0 nella sostituzione `E0_native + mean(E_ROKS_i − E0_i)`. Non sono medie di Hessiane rilassate indipendentemente. I controlli scalare/gradiente e Hessiana-vettore usano due passi, 2×10⁻⁴ e 10⁻⁴ bohr. I singoletti sono CSF con orbitali spaziali comuni ai due determinanti di ciascuna componente; non si attribuisce una proiezione esatta a un determinante broken-symmetry ottimizzato separatamente.',
'',
'È stato inoltre verificato indipendentemente ogni elemento della Hessiana differenziando tutte le componenti del gradiente:',
'', '| Sistema | Colonne | max errore Hessiana (Ha/bohr²) | max errore gradiente (Ha/bohr) |', '|---|---:|---:|---:|']
for r in columns:lines.append(f"| {r['molecule']} | {r['columns']} | {r['hessian_max_error']:.2e} | {r['scalar_gradient_max_error']:.2e} |")
lines += ['', '## Tempi sul medesimo host','',
'Calibrazione e stati S0 sono riutilizzati. Entrambi i metodi ricostruiscono i propri contesti nativi. Sono confrontati due spettri TDA completi con una singola energia ROKS già convergente, con due thread e tre ripetizioni; gradienti esclusi. Il costo per ottenere i due seed ROKS è escluso, quindi questi tempi favoriscono ROKS. Ogni matrice TDA ricostruita è confrontata con il checkpoint precedente per controllare kernel e orbitali.',
'', '| Molecola | Due TDA, tutti i root (s) | ROKS, uno stato caldo (s) |', '|---|---:|---:|']
for r in timing:lines.append(f"| {r['molecule']} | {r['tda_two_all_roots_seconds']:.4f} | {r['roks_one_state_warm_seconds']:.4f} |")
lines += ['',
'In questa implementazione ROKS non è più veloce per sole energie su queste piccole molecole. Il vantaggio verificato riguarda l’accuratezza media e l’accesso a una superficie con gradienti/Hessiane analitiche mediante energia, Fock, polarizzazione di spin e derivate del programma fondamentale GFN2, completate dal solver orbitale e dalla risposta KKT.',
'', '## Integrazione e riproducibilità','',
'LINK seleziona `native_roks_composite_orbital_coupling="relative_active_pair"`, `native_roks_composite_following="character_newton_index"`, `native_roks_composite_reference_policy="geometry_only"` e `native_roks_composite_ground_surface="native_plus_gaps"`. La prima esecuzione legge automaticamente i due `native_roks_composite_seed_paths`, con checksum e validazione del modello/promozione/indice. I restart successivi leggono la generazione accettata completa, comprendente vincoli, moltiplicatori, due CSF e tre riferimenti S0. Il metodo resta esplicitamente selezionabile e non modifica il default.',
'',
'Tutti i lavori elettronici, le ottimizzazioni e i benchmark sono stati eseguiti su **oracle.sns.it**, previa verifica dell’host. Il MacBook ha soltanto orchestrato i lavori, controllato i sorgenti e trasferito/letto i risultati. I checkpoint originali e le generazioni accettate restano disponibili sul server. Gli archivi locali selettivi omettono solo cache di calibrazione ricostruibili come `projected_eri.npz` e `quadrature_progress.npz`.',
'',
'Risultati numerici completi: [SUMMARY.json](SUMMARY.json). Sorgenti/protocollo: [PROTOCOL.json](PROTOCOL.json). Log e checkpoint verificabili: `remote/`. Il confronto storico a 15 stati è una regressione esposta, non una validazione cieca. La promozione a correzione universale richiede ulteriore validazione di accuratezza, in particolare oltre il sottospazio n→π* a singola eccitazione.']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:summary[k] for k in ['all_statistics','npi_statistics','prospective_statistics']},indent=2))
