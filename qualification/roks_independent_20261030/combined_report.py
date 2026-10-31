"""Report fixed combinations without confusing energy averaging with optimization."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import json
R=BASE/'results';d=json.loads((R/'combined_summary.json').read_text());s=d['statistics'];labels=['raw_full','spectral','energy_mean_50_50','mixed_functional_50_50']
names={'raw_full':'Risposta completa','spectral':'Profilo spettrale','energy_mean_50_50':'Media delle due energie, 50/50','mixed_functional_50_50':'Funzionale medio, riottimizzato 50/50'}
lines=['# Combinazione dei due modelli ROKS senza offset','',
'È stata verificata la proposta di combinare risposta completa e profilo spettrale. La media fissa delle energie riduce il MAE complessivo sui nove stati appaiati, ma non è equivalente a ottimizzare la media dei funzionali e non migliora ogni molecola. Nessun peso è stato adattato alle TBEs.',
'', '| Metodo | Stati | MAE / eV | MAE n→π* / eV (8 stati) | Errore massimo / eV |','|---|---:|---:|---:|---:|']
for label in labels:
 a=s['all'][label];n=s['n_pi'][label];lines.append(f"| {names[label]} | {a['count']} | {a['mae_ev']:.4f} | {n['mae_ev']:.4f} | {a['maxabs_ev']:.4f} |")
lines+=['', '## Risultati appaiati','', '| Molecola / stato | QUEST | Risposta completa | Spettrale | Media energie | Funzionale medio riottimizzato |', '|---|---:|---:|---:|---:|---:|']
for r in d['rows']:
 vals=[f"{r['models'][label]['excitation_ev']:.4f}" if r['models'][label].get('qualified') else 'non qualificato' for label in labels]
 lines.append(f"| {r['molecule']} {r['state']} | {r['tbe_ev']:.3f} | "+' | '.join(vals)+' |')
lines+=['','| Gruppo | N | Completo MAE | Spettrale MAE | Media energie MAE | Funzionale medio MAE |','|---|---:|---:|---:|---:|---:|']
for group in ['development','B_exposed','C_exposed']:
 x=s[group];lines.append(f"| {group} | {x['raw_full']['count']} | "+' | '.join(f"{x[l]['mae_ev']:.4f}" for l in labels)+' |')
lines+=['', 'Il gruppo C era indipendente quando è stato applicato il profilo spettrale congelato: MAE 0,1681 contro 0,3289 eV del completo e 1,1218 eV del ROKS nativo. I dati C erano già esposti quando sono state definite queste combinazioni: per esse non è una validazione cieca. Le combinazioni richiedono molecole nuove prima di essere promosse.',
'', '## Due combinazioni diverse','',
'**Media delle energie:** Ω̄=½Ω_completo+½Ω_spettrale. Le due CSF vengono ottimizzate separatamente e devono rappresentare lo stesso stato. È un modello composito a peso fisso, non la soluzione ROKS di un singolo funzionale medio. Non si aggiunge alcuna costante specifica di stato. Migliora la media ma l’errore massimo sale a 0,4019 eV (tetrazina), contro 0,2944 eV del solo spettrale.',
'',
'**Media dei funzionali:** Ē[Pα,Pβ]=½E_completo[Pα,Pβ]+½E_spettrale[Pα,Pβ], seguita da ottimizzazione separata di fondamentale ed eccitato. I due modelli condividono esattamente risposta di carica/spin e popolazione di riferimento, quindi si media soltanto il termine lineare della Fock: il kernel non viene raddoppiato. λ=0,9 resta applicato una sola volta allo scambio della risposta; non scala i gap. Il risultato non coincide con la media delle energie, per il rilassamento orbitale non lineare e per la struttura dei saddle.',
'',
'Il caso più evidente è uracile n→π*: la media delle energie è 4,8850 eV, mentre il funzionale medio riottimizzato dà 4,4520 eV. La stessa soluzione è stata raggiunta sia dalla partenza canonica sia da una continuazione indipendente; il confronto non sceglie la soluzione con errore più piccolo.',
'',
'È stata provata anche la combinazione senza peso libero che conserva la diagonale spettrale e recupera gli elementi non diagonali della Fock completa. Peggiora già i due n→π* di sviluppo: HNO 1,2032 contro 1,743 eV, uracile 4,2797 contro 4,935 eV. Questo ramo è stato scartato dopo i due fallimenti, conservando checkpoint e tentativi π→π* non convergenti. Non gli viene attribuita una statistica sui nove stati.',
'', '## TDA e derivate','',
'La TDA è stata calcolata per il funzionale medio sul suo stato fondamentale GFN2, con il medesimo kernel. Le TDA dei due componenti sono anch’esse su orbitali GFN2. La media a posteriori delle loro energie TDA, quando entrambe le assegnazioni passano il gate, è solo un controllo numerico composito; non viene presentata come TDA di un singolo funzionale ROKS.',
'', '| Molecola / stato | TDA completo | TDA spettrale | Media dei valori TDA | TDA funzionale medio |', '|---|---:|---:|---:|---:|']
for r in d['rows']:
 vals=[];energies=[]
 for folder in [R/'full_response',R/'qualified_spectral_response',R/'combined_response/mixed_functional_50_50']:
  p=folder/r['molecule']/'tda.json';rr=next((x for x in json.loads(p.read_text())['rows'] if x['state']==r['state']),None) if p.exists() else None
  e=rr['energy_ev'] if rr and rr.get('qualified') else None;energies.append(e)
 vals=[f'{e:.4f}' if e is not None else 'non qualificata' for e in energies]
 mid=f'{.5*(energies[0]+energies[1]):.4f}' if energies[0] is not None and energies[1] is not None else 'non qualificata'
 lines.append(f"| {r['molecule']} {r['state']} | {vals[0]} | {vals[1]} | {mid} | {vals[2]} |")
lines+=['',
'Con pesi costanti, le derivate di una superficie composita sarebbero le medie delle derivate dei componenti: ḡ=½(g₁+g₂), H̄=½(H₁+H₂). Qui i due nuovi modelli non locali restano solo verticali: queste Hessiane nucleari dei componenti non sono ancora implementate/validate, quindi non vengono fornite come se esistessero. Le Hessiane analitiche del candidato shell precedente restano quelle già documentate e non si trasferiscono automaticamente a questi modelli.',
'', '## Scelta del prossimo candidato','',
'La media delle energie con peso fissato 1/2 è il candidato composito più promettente per il MAE n→π* in questo confronto. Il profilo spettrale mantiene il miglior errore massimo. Il funzionale medio non offre un vantaggio netto sullo spettrale e il recupero dei soli accoppiamenti non diagonali è scartato. Nessuna combinazione è dichiarata una correzione universale, né sostituisce il metodo predefinito.',
'', 'Tutti i nuovi lavori elettronici sono stati eseguiti su oracle.sns.it verificato. Il solver del riferimento TDA della pirazina è stato ripreso dal checkpoint dello stesso autostato: 1152 prodotti matrice-vettore completati conservati, SCF non ripetuto, residuo finale 1,22×10⁻¹⁰ senza cambiare root, operatore o tolleranza richiesta. Protocollo delle combinazioni e provenienza sono in results/combined_response/protocol.json e results/combined_provenance.json.']
(BASE/'COMBINATION_REPORT_20261030.md').write_text('\n'.join(lines)+'\n');print('COMBINATION REPORT',flush=True)
