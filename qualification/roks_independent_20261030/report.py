"""Report the independent candidate separately from subsequent development."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
R=BASE/'results';d=json.loads((R/'cohort_summary.json').read_text());rows=d['rows'];stats=d['statistics']
lines=['# ROKS senza offset: insieme indipendente e diagnosi n→π*','',
'Il candidato a risposta di carica/spin corretta è stato implementato come funzionale autonomo, salvabile e utilizzabile da LINK senza ancoraggi energetici. Sette molecole nuove forniscono otto stati qualificati. L’accuratezza migliora rispetto al ROKS originale, ma l’errore n→π* non è risolto universalmente.',
'', '## Prova indipendente','',
'Protocollo fissato prima delle previsioni del candidato: otto molecole, 13 stati valenza singoletti sicuri QUEST sotto 8 eV. Assegnazioni indipendenti archiviate riutilizzate; overlap di proiezione, promozione dominante e CSF ≥0,7. Nessuna sostituzione di stati o modifica delle soglie per migliorare le statistiche. I vecchi 13 stati non sono inclusi in queste medie. Le otto nuove molecole non hanno calibrato i parametri del candidato.',
'', '| Insieme appaiato | N | ROKS originale MAE / eV | ROKS corretto MAE / eV | TDA stesso funzionale MAE / eV |', '|---|---:|---:|---:|---:|']
for family,label in [('all','Tutti'),('n_pi','n→π*'),('pi_pi','π→π*')]:
 s=stats[family];lines.append(f"| {label} | {s['corrected']['count']} | {s['native']['mae_ev']:.4f} | {s['corrected']['mae_ev']:.4f} | {s['tda_paired']['mae_ev']:.4f} |")
lines+=['', '| Molecola | Stato | QUEST | ROKS originale | ROKS corretto | Errore corretto | TDA |', '|---|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['molecule']} | {r['state']} | {r['tbe_ev']:.3f} | {r['native_roks_ev']:.4f} | {r['excitation_ev']:.4f} | {r['error_ev']:+.4f} | {r['tda']['energy_ev']:.4f} |")
lines+=['',f"Copertura: {d['qualified_states']}/{d['planned_states']} stati, {d['qualified_molecules']}/{d['planned_molecules']} molecole. Tutti gli otto errori corretti sono negativi: il bias residuo non viene nascosto aggiungendo una costante. L’errore massimo n→π* resta {stats['n_pi']['corrected']['maxabs_ev']:.4f} eV.",
'', 'Stati esclusi:']
for e in d['exclusions']:lines.append(f"- {e['molecule']} {e['state']}: {e['error']}.")
lines+=['',
'La normalizzazione della notazione Cs A doppio-primo verso il nome usato da PySCF è stata corretta prima della selezione finale. I tentativi precedenti sono conservati. Questo è un adattamento della notazione, non un cambio di simmetria o di root. Il protocol_selection_draft.json documenta il primo elenco; protocol.json ha fissato le assegnazioni già archiviate prima delle previsioni del modello.',
'', '## Confronto TDA e costo','',
'La TDA usa gli orbitali di stato fondamentale GFN2 del medesimo funzionale corretto e il suo kernel singoletto; nessun orbitale DFT sostituisce la base delle transizioni. Il contributo di spin si annulla per la perturbazione singoletta chiusa, mentre la risposta di carica corretta entra nel kernel. I file tda.npz conservano le matrici e i vettori completi. Questo è il confronto variazionale SCC/ROKS: non è il diverso kernel empirico Ohno/sTDA.',
'', '| Molecola / stato | ROKS avvio canonico (s) | TDA costruzione + tutte le radici (s) |', '|---|---:|---:|']
for r in rows:
 td=json.loads((R/'predictions'/r['molecule']/'shell_charge_spin/tda.json').read_text())
 lines.append(f"| {r['molecule']} {r['state']} | {r['seconds_current_run']:.4f} | {td['solver_seconds']:.4f} |")
lines+=['',
'ROKS include la soluzione dello stato fondamentale con il medesimo funzionale; TDA riporta separatamente ground_seconds. La calibrazione DFT della risposta è un costo aggiuntivo e si trova in shell_calibration.json. I tempi non sostengono una superiorità di velocità di ROKS; TDA restituisce più radici per matrice e ROKS ottimizza uno stato. Nessun rilancio di SCF DFT fondamentale è stato necessario per questo insieme: i riferimenti già completati sono stati riutilizzati.',
'', '## Funzionale, gradienti e Hessiane','',
'Il candidato shell aggiunge E_c=½(n−n₀)ᵀU(n−n₀) e usa E_spin=½mᵀWm. W, U e n₀ sono congelati per la superficie locale; metrica AO e densità variano. Non esiste un offset energetico dello stato eccitato. ROKS ottimizza la CSF singoletto a orbitali comuni con E_S=2E_misto−E_tripletto. Il programma di stato fondamentale fornisce energie, Fock e risposta; GMF cerca il saddle elettronico. L’energia fondamentale è ricalcolata con lo stesso funzionale, evitando gap riferiti a due Hamiltoniane diverse.',
'',
'Il fattore uracile λ=0,9 agisce una sola volta sullo scambio HF nella risposta PBE0, mantenendo invariato il kernel semilocale. Non è una moltiplicazione dei gap o dell’energia finale. La Fock di riferimento degli esperimenti di primo ordine è quella PBE0, senza una seconda applicazione di λ.',
'',
'Per ogni stato indipendente qualificato, results/candidate/MOLECOLA/STATO contiene checkpoint, Hessiana nucleare analitica, validazione direzionale indipendente e backend.json. Le derivate sono quelle del funzionale corretto congelato, non quelle DFT. Il metodo analitico non usa geometrie spostate; i punti ±2×10⁻⁴ bohr servono soltanto alla verifica. Indice orbitale, risposta, invarianza per traslazione e confronto con differenze di gradienti vengono controllati esplicitamente.',
'']
validations=[json.loads(p.read_text()) for p in (R/'candidate').glob('*/*/validation.json')]
if validations:
 lines+= [f"Superfici verificate: {len(validations)}, tutte passate: {all(v['passed'] for v in validations)}. Massimo errore direzionale di gradiente {max(v['gradient_direction_error'] for v in validations):.3e} Ha/bohr; di Hessiana-vettore {max(v['hessian_vector_error'] for v in validations):.3e} Ha/bohr².", '']
lines+=['La nuova API persiste la risposta con save_response_reference/load_response_reference. LINK accetta native_roks_response_path, verifica checksum/fingerprint e lo include nell’identità del restart; un modello senza offset non può essere combinato accidentalmente con native_roks_anchor_path. Il controllo molecolare reale LINK/APOC riproduce energia, gradiente e Hessiana diretti con differenze sotto 10⁻⁸, senza ancoraggio energetico.',
'', '## Varianti sviluppate e scartate','',
'La correzione universale non può essere dichiarata risolta selezionando un λ per molecola. Sono state costruite e controllate due estensioni del funzionale, poi scartate sulla base di HNO e uracile, prima di usarle per scegliere un profilo sui nuovi stati:',
'',
'- Tensori locali: componenti sp di rango 1 e pp a traccia nulla di rango 2 della matrice simmetrizzata P S. Una costante comune per ogni blocco irriducibile garantisce invarianza alle rotazioni. Le costanti derivano dalla traccia della risposta DFT proiettata su densità duali, senza usare le TBEs. Sono state provate spin e carica+spin. Le Hessiane analitiche sono verificate su tutte le nove coordinate di HNO; la chimica peggiora nettamente. Non si usa questa variante per migliorare artificialmente il risultato del test indipendente.',
'- Primo ordine sui gusci: potenziale lineare variazionale proiettato dalla differenza fra Fock DFT e GFN2, con gauge uniforme rimosso. Lo stato fondamentale viene rilassato nello stesso potenziale. Anche questa variante peggiora HNO e uracile; la proiezione scalare lascia gran parte del residuo non locale. Non è un offset dello stato eccitato.',
'', '| Variante di sviluppo | Molecola / stato | Energia / eV | Errore / eV |', '|---|---|---:|---:|']
for name in ['nitroxyl','uracil']:
 for label in ['shell_charge_spin','tensor_spin','tensor_charge_spin','fock_shell_charge_spin']:
  for p in sorted((R/'predictions'/name/label).glob('*/result.json')):
   row=json.loads(p.read_text())
   if row.get('qualified'):lines.append(f"| {label} | {name} {row['state']} | {row['excitation_ev']:.4f} | {row['error_ev']:+.4f} |")
lines+=['', 'La ricerca di ulteriori indici per uracile π→π* con tensori carica+spin è stata interrotta dopo il fallimento già decisivo n→π*, conservando i checkpoint e i tentativi con identità sbagliata. Non è riportata come previsione qualificata.',
'', '## Risposta completa non locale: diagnostica successiva','',
'È stato costruito un prototipo verticale che conserva tutte le coppie simmetriche AO: Fock proiettata completa, risposta di carica DFT meno quella GFN2 già presente, e risposta completa di spin DFT. Include la densità di core DFT congelata e la densità di valenza GFN2 incorporata isometricamente nella base DFT. La risposta XC GGA viene integrata direttamente sulle coppie AO, con integrali Coulomb/scambio trasformati nella base compatta. Non si ottimizzano parametri sulle energie eccitate.',
'',
'È una diagnosi sviluppata dopo la prova indipendente, non un nuovo successo cieco. Il modello completo resta solo verticale: i gradienti e le Hessiane nucleari vengono rifiutati esplicitamente. Le Hessiane analitiche del candidato shell e dei modelli locali testati non vengono attribuite a questo prototipo.',
'', '| Molecola / stato | Risposta completa / eV | Errore / eV | CSF overlap |', '|---|---:|---:|---:|']
for p in sorted((R/'full_response').glob('*/*/result.json')):
 r=json.loads(p.read_text())
 if r.get('qualified'):lines.append(f"| {r['molecule']} {r['state']} | {r['excitation_ev']:.4f} | {r['error_ev']:+.4f} | {r['csf_overlap']:.4f} |")
for p in sorted((R/'full_response').glob('*/independent_audit.json')):
 a=json.loads(p.read_text());lines+=['',f"Audit indipendente {a['molecule']}: passato={a['passed']}; errore massimo della risposta contro gen_response UKS PySCF {max(max(c.values()) for c in a['checks']):.3e} Ha; errore energia/Fock {a['energy_fock_direction_error']:.3e} Ha."]
btest=json.loads((R/'holdout_B_summary.json').read_text())
lines+=['', '## Secondo test cieco: tre carbonili nuovi','',
'Il profilo completo e λ=0,9 sono stati congelati prima di calcolare acetaldeide, acetone e formammide. Geometrie e TBEs provengono da [QUESTDB](https://github.com/pfloos/QUESTDB), commit 460482bcc6231dab41fe32188dc43959a7c2bdaa. Root DFT TDA 1 fissata, senza sostituzioni. Soglie di identità e criteri di successo sono registrati in full_holdout_protocol.json. Nessun parametro è stato aggiustato sulle tre energie.',
'', '| Molecola | QUEST | ROKS originale | Shell senza offset | Risposta completa |', '|---|---:|---:|---:|---:|']
for row in btest['rows']:
 lines.append(f"| {row['molecule']} | {row['tbe_ev']:.3f} | {row['native_roks_ev']:.4f} | {row['shell_corrected_roks_ev']:.4f} | {row['full_response']['excitation_ev']:.4f} |")
lines+=['',f"MAE sui tre n→π*: originale {btest['mae_ev']['native']:.4f}, shell {btest['mae_ev']['shell']:.4f}, completo {btest['mae_ev']['full']:.4f} eV. Criterio predefinito superato: {btest['passed']}. Tutti gli stati sono qualificati, ma la risposta completa peggiora la media rispetto a entrambi i controlli. Non può essere promossa come soluzione universale.",
'', '## Controlli conclusivi','',
'La raffinazione della verifica Hessiana-vettore per piridazina π→π* da 2×10⁻⁴ a 10⁻⁴ e 5×10⁻⁵ bohr riduce il residuo da 3,22×10⁻⁵ a 8,05×10⁻⁶ e 2,01×10⁻⁶ Ha/bohr², coerentemente con il troncamento quadratico delle differenze centrali.',
'',
'La diagnosi non lineare coerente con lo stesso λ cambia le energie verticali del modello completo, a orbitali mantenuti fissi, di +0,0242 eV in HNO e −0,0451/+0,0024 eV in uracile n→π*/π→π*. Non sono nuovi stati stazionari né correzioni da sommare alle previsioni: mostrano che i termini superiori da soli non spiegano il grande errore π→π*.',
'',
'Anche le due maschere di sviluppo che conservano i blocchi di spin GFN2 sullo stesso atomo e aggiungono soltanto la risposta interatomica sono archiviate in nonlocal_screen_summary.json. Sono prove sui dieci vecchi stati di sviluppo, separate dai due test indipendenti; non giustificano una correzione universale.',
'']
lines+=['', '## Decisione','',

'Il candidato senza offset è distribuibile come modalità sperimentale e migliora l’accuratezza media sull’insieme indipendente più ampio. Non diventa il metodo predefinito: restano errori n→π* grandi e un bias sistematico. I modelli tensoriali locali e la sola correzione scalare della Fock non risolvono il problema; le prove negative sono conservate. Nessuna correzione universale n→π* è rivendicata.',
'', '## Riferimenti e tracciabilità','',
'La forma standard a popolazioni di guscio e la derivazione delle costanti di spin sono discusse da [Neugebauer et al., JCC 2023](https://doi.org/10.1002/jcc.27185) e [Moradi et al., JCC 2024](https://doi.org/10.1002/jcc.27482). I nuovi termini e i loro esiti sono esperimenti di questa implementazione, non risultati rivendicati da quei lavori.',
'',
'Tutti i calcoli elettronici e numerici sono stati eseguiti su oracle.sns.it, dopo verifica del nome host. Il laptop ha svolto orchestrazione, editing e verifiche di file. Source e checkpoint precedenti, colonne completate della risposta e riferimenti SCF DFT sono conservati e riutilizzati. provenance.json registra gli hash di input, script, modelli, risultati e ambiente. tests_final.log documenta 47 test passati; i controlli molecolari e delle derivate sono salvati separatamente.']
(BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('REPORT',len(rows),'independent states;',len(validations),'derivative validations')
