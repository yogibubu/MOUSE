"""Report new diagnostics separately from the immutable previous campaign."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import json,sys
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import digest
R=BASE/'results';A=R/'response_ablation';Q=R/'qualified_spectral_response'
ab=json.loads((A/'summary.json').read_text());dc=json.loads((A/'decomposition.json').read_text());st=json.loads((A/'ground_stability.json').read_text());parent=json.loads((A/'dft_ground_stability.json').read_text());C=json.loads((R/'holdout_C_summary.json').read_text())
dev=json.loads((Q/'summary.json').read_text())['rows'];B=json.loads((Q/'carbonyl_summary.json').read_text())['rows'];S=dev+B
lines=['# Ripresa ROKS: diagnosi della risposta e nuova prova indipendente','',
'Esperimenti del 30 ottobre 2026, interamente su oracle.sns.it. Si riutilizzano SCF DFT, modelli di risposta e checkpoint precedenti. Nessun offset di stato e nessun adattamento di λ alle energie eccitate. Il rapporto precedente REPORT.md e la sua provenienza restano separati.',
'', '## Separazione dei termini','',
'Sette ablation del medesimo funzionale, oltre al modello completo già calcolato, su HNO, due stati dell’uracile e i tre carbonili già esposti. Le tre componenti sono il termine lineare della Fock, la risposta di carica e la sostituzione della risposta di spin. Lo spin GFN2 nativo viene ricostruito esattamente nella base di coppie AO e controllato in energia e Fock prima degli esperimenti. Non si sceglie un profilo sulla base dei nuovi risultati.',
'', '| Profilo | Stati | MAE / eV | MAE n→π* / eV |', '|---|---:|---:|---:|']
for label,s in ab['statistics'].items():lines.append(f"| {label} | {s['count']} | {s['mae_ev']:.4f} | {s['npi_mae_ev']:.4f} |")
lines+=['',f"Le {len(dc['rows'])} energie vengono ricostruite dalla somma dei contributi con errore massimo {max(abs(r['closure_error_ev']) for r in dc['rows']):.3e} eV. La decomposizione non modifica le energie e non rappresenta una correzione da sommare alle previsioni.",
'', '| Stato, modello completo | GFN2 sugli orbitali del modello | Fock | Carica | Sostituzione spin | Totale |', '|---|---:|---:|---:|---:|---:|']
for r in dc['rows']:
 if r['variant']=='full':lines.append(f"| {r['molecule']} {r['state']} | {r['native_on_model_orbitals_ev']:.4f} | {r['reference_fock_ev']:+.4f} | {r['charge_response_ev']:+.4f} | {r['spin_replacement_ev']:+.4f} | {r['excitation_ev']:.4f} |")
lines+=['', 'I termini trasferiti si compensano per diversi eV. La sola correttezza delle derivate non garantisce accuratezza delle eccitazioni; trasferire soltanto carica o spin può produrre energie negative rispetto al fondamentale chiuso. Questi risultati numericamente stazionari sono conservati nella tabella, senza rimuoverli dalle statistiche delle ablation.',
'', '## Stabilità elettronica','',
f"Verificate {len(st['rows'])} curvature elettroniche reali singoletto/tripletto con differenze centrali di energia: errore massimo {max(r['error_hartree'] for r in st['rows']):.3e} Ha. Non sono Hessiane nucleari. In HNO il modello completo ha un modo tripletto negativo (autovalore di H/4 = −0,020353 Ha). La variante spettrale sotto descritta conserva un modo negativo (−0,013997 Ha).",
'',f"Il riferimento PBE0 completo, riletto dal checkpoint senza ripetere SCF, è anch’esso instabile: H/4 = {parent['lowest_triplet_hessian_over4_hartree']:.8f} Ha, residuo Davidson {parent['residual']:.3e}. Questo controllo usa PBE0 a=0,25, mentre il modello compatto usa λ=0,9 nella risposta. L’instabilità non viene quindi attribuita soltanto alla proiezione GFN2 e non viene rimossa con un’aggiunta empirica.",
'', '## Trasporto delle energie orbitali DFT','',
'Il trasporto ortogonale dell’intero spettro viene rifiutato prima delle previsioni: gli spazi virtuali selezionati hanno valori singolari minimi 0,4925 in HNO e 0,0443 nell’uracile, sotto la soglia fissata 0,5. Il pivoting QR non risolve il problema dell’uracile (0,1687). Le singole coppie n/π* e π/π* di interesse hanno invece mapping affidabile.',
'',
'È stato quindi costruito un unico profilo distinto, qualified-spectral-full-response: termine lineare diagonale nella base canonica GFN2, livelli DFT trasferiti solo con overlap ≥0,7 e margine ≥0,05, energie non qualificate mantenute native nella stessa gauge del centroide occupato. Il termine è parte del funzionale variazionale e cambia la Fock; non è una costante aggiunta all’energia di uno stato. La risposta completa di carica/spin resta identica. λ=0,9 agisce una sola volta sullo scambio della risposta; i gap DFT non vengono moltiplicati per λ.',
'', '| Dati già esposti | QUEST | Profilo spettrale | Errore |', '|---|---:|---:|---:|']
for r in S:lines.append(f"| {r['molecule']} {r['state']} | {r['tbe_ev']:.3f} | {r['excitation_ev']:.4f} | {r['error_ev']:+.4f} |")
for label,rr in [('Tre stati di sviluppo',dev),('Tre carbonili già esposti',B),('Tutti i cinque n→π* esposti',[r for r in S if r['state']=='n_pi'])]:
 lines+=['',f"{label}: MAE {sum(abs(r['error_ev']) for r in rr)/len(rr):.4f} eV."]
lines+=['', 'Il miglioramento dell’uracile π→π* si accompagna a un peggioramento sistematico dei carbonili. Questi ultimi non sono una nuova prova cieca: i loro risultati precedenti erano già noti. Il profilo resta invariato nel successivo test C.',
'', '## Tre molecole indipendenti: insieme C','',
f"Protocollo congelato prima delle previsioni, SHA256 {C['protocol_sha256']}. Gliossale, pirazina e tetrazina non hanno selezionato il modello. Primo stato n→π* singoletto valenza sicuro QUEST; root DFT TDA 1 fissata senza sostituzioni, proiezione e identità ≥0,7, residuo orbitale ≤10⁻⁹. Tutti e tre gli stati sono richiesti dal criterio di successo.",
'', '| Molecola | QUEST | ROKS nativo | Risposta completa precedente | Profilo spettrale | TDA stesso funzionale/GFN2 |', '|---|---:|---:|---:|---:|---:|']
for r in C['rows']:
 if r.get('paired_qualified'):
  td=next((x for x in (r['tda'] or {}).get('rows',[]) if x['state']=='n_pi'),None);tv=f"{td['energy_ev']:.4f}" if td and td['qualified'] else 'non qualificata'
  lines.append(f"| {r['molecule']} | {r['tbe_ev']:.3f} | {r['native_roks_ev']:.4f} | {r['raw_full_response']['excitation_ev']:.4f} | {r['spectral_response']['excitation_ev']:.4f} | {tv} |")
 else:lines.append(f"| {r['molecule']} | — | — | — | escluso: identità/soluzione non qualificata | — |")
for label,s in C['statistics'].items():lines+=['',f"{label}: N={s['count']}, MAE={s['mae_ev']}, massimo assoluto={s['maxabs_ev']} eV."]
lines+=['',f"Criterio indipendente superato: **{C['passed']}**. Il profilo non viene modificato dopo aver osservato C.",
'', 'La TDA usa gli orbitali GFN2 ottenuti con lo stesso funzionale del profilo spettrale, con la stessa risposta di carica. Gli orbitali DFT servono alla calibrazione e all’assegnazione di riferimento, non sostituiscono la base delle transizioni TDA. Tempi di soluzione ROKS e TDA, e costo separato della preparazione DFT, sono nei JSON/NPZ. Non si rivendica un vantaggio generale di tempo.',
'', '## Continuità e limiti delle derivate','',
'Le partenze canoniche possono convergere a saddle di diversa identità: questi tentativi e i relativi checkpoint non vengono eliminati. La continuazione della Fock completa verso quella spettrale registra overlap tra CSF consecutive e rispetto alla promozione originale. Passaggi intermedi t<1 sono diagnostiche numeriche, non nuovi parametri selezionabili o previsioni del profilo finale. I tentativi che perdono l’identità originale restano esclusi.',
'', 'Questo prototipo non locale è **solo verticale**. Non gli vengono attribuite le Hessiane nucleari analitiche già validate per il candidato shell. I 47 test precedenti e le otto superfici con Hessiana analitica rimangono quelli documentati in REPORT.md; qui sono aggiunti controlli elettronici indipendenti, non una nuova implementazione di derivate nucleari.',
'', '## Esito','',
'La correzione universale n→π* resta irrisolta: il profilo spettrale riduce l’errore su alcuni stati ma peggiora i carbonili già esposti. I nuovi risultati distinguono errore del funzionale, instabilità del riferimento e perdita di identità dello stato. Non si introducono offset, λ per molecola o correzioni scelte sulle TBEs. Nessun metodo predefinito è cambiato.',
'', 'Dati molecolari: [QUESTDB](https://github.com/pfloos/QUESTDB), commit 460482bcc6231dab41fe32188dc43959a7c2bdaa. Metadati, geometrie, licenza, checkpoint, script eseguiti e risultati sono conservati; resumed_provenance.json registra gli hash della ripresa separatamente da provenance.json della campagna precedente.']
(BASE/'RESUMED_REPORT_20261030.md').write_text('\n'.join(lines)+'\n')
print('RESUMED REPORT',C['passed'],flush=True)
