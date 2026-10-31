# ROKS senza offset: insieme indipendente e diagnosi n→π*

Il candidato a risposta di carica/spin corretta è stato implementato come funzionale autonomo, salvabile e utilizzabile da LINK senza ancoraggi energetici. Sette molecole nuove forniscono otto stati qualificati. L’accuratezza migliora rispetto al ROKS originale, ma l’errore n→π* non è risolto universalmente.

## Prova indipendente

Protocollo fissato prima delle previsioni del candidato: otto molecole, 13 stati valenza singoletti sicuri QUEST sotto 8 eV. Assegnazioni indipendenti archiviate riutilizzate; overlap di proiezione, promozione dominante e CSF ≥0,7. Nessuna sostituzione di stati o modifica delle soglie per migliorare le statistiche. I vecchi 13 stati non sono inclusi in queste medie. Le otto nuove molecole non hanno calibrato i parametri del candidato.

| Insieme appaiato | N | ROKS originale MAE / eV | ROKS corretto MAE / eV | TDA stesso funzionale MAE / eV |
|---|---:|---:|---:|---:|
| Tutti | 8 | 0.4829 | 0.3697 | 0.8769 |
| n→π* | 5 | 0.6221 | 0.4738 | 1.2019 |
| π→π* | 3 | 0.2509 | 0.1961 | 0.3352 |

| Molecola | Stato | QUEST | ROKS originale | ROKS corretto | Errore corretto | TDA |
|---|---|---:|---:|---:|---:|---:|
| cyanoformaldehyde | n_pi | 3.817 | 3.0433 | 3.3204 | -0.4966 | 2.6932 |
| cyclopropenone | n_pi | 4.274 | 4.0470 | 4.1190 | -0.1550 | 3.2562 |
| methylenecyclopropene | pi_pi | 4.280 | 4.3634 | 4.2593 | -0.0207 | 3.8768 |
| pyridazine | n_pi | 3.830 | 2.8818 | 3.1029 | -0.7271 | 2.4296 |
| pyridazine | pi_pi | 5.245 | 4.8605 | 4.9465 | -0.2985 | 5.0864 |
| nitrobenzene | n_pi | 3.926 | 3.7299 | 3.9079 | -0.0181 | 3.3957 |
| benzonitrile | pi_pi | 4.864 | 4.5791 | 4.5949 | -0.2691 | 4.4202 |
| imidazole | n_pi | 6.715 | 5.7493 | 5.7427 | -0.9723 | 4.7778 |

Copertura: 8/13 stati, 7/8 molecole. Tutti gli otto errori corretti sono negativi: il bias residuo non viene nascosto aggiungendo una costante. L’errore massimo n→π* resta 0.9723 eV.

Stati esclusi:
- cyanoformaldehyde pi_pi: no stationary native seed passes fixed identity rules.
- cyclopropenone pi_pi: archived family assignment unresolved.
- aniline pi_pi: fixed reference root fails symmetry or compact projection.
- nitrobenzene pi_pi: no stationary native seed passes fixed identity rules.
- imidazole pi_pi: archived family assignment unresolved.

La normalizzazione della notazione Cs A doppio-primo verso il nome usato da PySCF è stata corretta prima della selezione finale. I tentativi precedenti sono conservati. Questo è un adattamento della notazione, non un cambio di simmetria o di root. Il protocol_selection_draft.json documenta il primo elenco; protocol.json ha fissato le assegnazioni già archiviate prima delle previsioni del modello.

## Confronto TDA e costo

La TDA usa gli orbitali di stato fondamentale GFN2 del medesimo funzionale corretto e il suo kernel singoletto; nessun orbitale DFT sostituisce la base delle transizioni. Il contributo di spin si annulla per la perturbazione singoletta chiusa, mentre la risposta di carica corretta entra nel kernel. I file tda.npz conservano le matrici e i vettori completi. Questo è il confronto variazionale SCC/ROKS: non è il diverso kernel empirico Ohno/sTDA.

| Molecola / stato | ROKS avvio canonico (s) | TDA costruzione + tutte le radici (s) |
|---|---:|---:|
| cyanoformaldehyde n_pi | 0.4245 | 0.0101 |
| cyclopropenone n_pi | 0.6707 | 0.0129 |
| methylenecyclopropene pi_pi | 0.7702 | 0.0182 |
| pyridazine n_pi | 1.5453 | 0.0704 |
| pyridazine pi_pi | 2.7694 | 0.0704 |
| nitrobenzene n_pi | 4.1838 | 0.4687 |
| benzonitrile pi_pi | 3.1061 | 0.2824 |
| imidazole n_pi | 1.0284 | 0.0327 |

ROKS include la soluzione dello stato fondamentale con il medesimo funzionale; TDA riporta separatamente ground_seconds. La calibrazione DFT della risposta è un costo aggiuntivo e si trova in shell_calibration.json. I tempi non sostengono una superiorità di velocità di ROKS; TDA restituisce più radici per matrice e ROKS ottimizza uno stato. Nessun rilancio di SCF DFT fondamentale è stato necessario per questo insieme: i riferimenti già completati sono stati riutilizzati.

## Funzionale, gradienti e Hessiane

Il candidato shell aggiunge E_c=½(n−n₀)ᵀU(n−n₀) e usa E_spin=½mᵀWm. W, U e n₀ sono congelati per la superficie locale; metrica AO e densità variano. Non esiste un offset energetico dello stato eccitato. ROKS ottimizza la CSF singoletto a orbitali comuni con E_S=2E_misto−E_tripletto. Il programma di stato fondamentale fornisce energie, Fock e risposta; GMF cerca il saddle elettronico. L’energia fondamentale è ricalcolata con lo stesso funzionale, evitando gap riferiti a due Hamiltoniane diverse.

Il fattore uracile λ=0,9 agisce una sola volta sullo scambio HF nella risposta PBE0, mantenendo invariato il kernel semilocale. Non è una moltiplicazione dei gap o dell’energia finale. La Fock di riferimento degli esperimenti di primo ordine è quella PBE0, senza una seconda applicazione di λ.

Per ogni stato indipendente qualificato, results/candidate/MOLECOLA/STATO contiene checkpoint, Hessiana nucleare analitica, validazione direzionale indipendente e backend.json. Le derivate sono quelle del funzionale corretto congelato, non quelle DFT. Il metodo analitico non usa geometrie spostate; i punti ±2×10⁻⁴ bohr servono soltanto alla verifica. Indice orbitale, risposta, invarianza per traslazione e confronto con differenze di gradienti vengono controllati esplicitamente.

Superfici verificate: 8, tutte passate: False. Massimo errore direzionale di gradiente 5.939e-08 Ha/bohr; di Hessiana-vettore 1.214e-04 Ha/bohr².

La nuova API persiste la risposta con save_response_reference/load_response_reference. LINK accetta native_roks_response_path, verifica checksum/fingerprint e lo include nell’identità del restart; un modello senza offset non può essere combinato accidentalmente con native_roks_anchor_path. Il controllo molecolare reale LINK/APOC riproduce energia, gradiente e Hessiana diretti con differenze sotto 10⁻⁸, senza ancoraggio energetico.

## Varianti sviluppate e scartate

La correzione universale non può essere dichiarata risolta selezionando un λ per molecola. Sono state costruite e controllate due estensioni del funzionale, poi scartate sulla base di HNO e uracile, prima di usarle per scegliere un profilo sui nuovi stati:

- Tensori locali: componenti sp di rango 1 e pp a traccia nulla di rango 2 della matrice simmetrizzata P S. Una costante comune per ogni blocco irriducibile garantisce invarianza alle rotazioni. Le costanti derivano dalla traccia della risposta DFT proiettata su densità duali, senza usare le TBEs. Sono state provate spin e carica+spin. Le Hessiane analitiche sono verificate su tutte le nove coordinate di HNO; la chimica peggiora nettamente. Non si usa questa variante per migliorare artificialmente il risultato del test indipendente.
- Primo ordine sui gusci: potenziale lineare variazionale proiettato dalla differenza fra Fock DFT e GFN2, con gauge uniforme rimosso. Lo stato fondamentale viene rilassato nello stesso potenziale. Anche questa variante peggiora HNO e uracile; la proiezione scalare lascia gran parte del residuo non locale. Non è un offset dello stato eccitato.

| Variante di sviluppo | Molecola / stato | Energia / eV | Errore / eV |
|---|---|---:|---:|
| shell_charge_spin | nitroxyl n_pi | 2.3115 | +0.5685 |
| tensor_spin | nitroxyl n_pi | 0.0783 | -1.6647 |
| tensor_charge_spin | nitroxyl n_pi | 0.0724 | -1.6706 |
| fock_shell_charge_spin | nitroxyl n_pi | 2.5859 | +0.8429 |
| shell_charge_spin | uracil n_pi | 4.3277 | -0.6073 |
| shell_charge_spin | uracil pi_pi | 4.4674 | -0.9136 |
| tensor_spin | uracil n_pi | 3.3561 | -1.5789 |
| tensor_spin | uracil pi_pi | 4.5640 | -0.8170 |
| tensor_charge_spin | uracil n_pi | 3.0012 | -1.9338 |
| fock_shell_charge_spin | uracil n_pi | 4.0768 | -0.8582 |
| fock_shell_charge_spin | uracil pi_pi | 4.3282 | -1.0528 |

La ricerca di ulteriori indici per uracile π→π* con tensori carica+spin è stata interrotta dopo il fallimento già decisivo n→π*, conservando i checkpoint e i tentativi con identità sbagliata. Non è riportata come previsione qualificata.

## Risposta completa non locale: diagnostica successiva

È stato costruito un prototipo verticale che conserva tutte le coppie simmetriche AO: Fock proiettata completa, risposta di carica DFT meno quella GFN2 già presente, e risposta completa di spin DFT. Include la densità di core DFT congelata e la densità di valenza GFN2 incorporata isometricamente nella base DFT. La risposta XC GGA viene integrata direttamente sulle coppie AO, con integrali Coulomb/scambio trasformati nella base compatta. Non si ottimizzano parametri sulle energie eccitate.

È una diagnosi sviluppata dopo la prova indipendente, non un nuovo successo cieco. Il modello completo resta solo verticale: i gradienti e le Hessiane nucleari vengono rifiutati esplicitamente. Le Hessiane analitiche del candidato shell e dei modelli locali testati non vengono attribuite a questo prototipo.

| Molecola / stato | Risposta completa / eV | Errore / eV | CSF overlap |
|---|---:|---:|---:|
| acetaldehyde n_pi | 4.5151 | +0.2111 | 0.9374 |
| acetone n_pi | 4.5025 | +0.0345 | 0.9412 |
| formamide n_pi | 5.8807 | +0.2367 | 0.9481 |
| nitroxyl n_pi | 1.4558 | -0.2872 | 0.9742 |
| uracil n_pi | 4.9954 | +0.0604 | 0.8798 |
| uracil pi_pi | 6.0503 | +0.6693 | 0.9071 |

Audit indipendente nitroxyl: passato=True; errore massimo della risposta contro gen_response UKS PySCF 1.575e-15 Ha; errore energia/Fock 1.051e-11 Ha.

Audit indipendente uracil: passato=True; errore massimo della risposta contro gen_response UKS PySCF 1.022e-14 Ha; errore energia/Fock 4.798e-10 Ha.

## Secondo test cieco: tre carbonili nuovi

Il profilo completo e λ=0,9 sono stati congelati prima di calcolare acetaldeide, acetone e formammide. Geometrie e TBEs provengono da [QUESTDB](https://github.com/pfloos/QUESTDB), commit 460482bcc6231dab41fe32188dc43959a7c2bdaa. Root DFT TDA 1 fissata, senza sostituzioni. Soglie di identità e criteri di successo sono registrati in full_holdout_protocol.json. Nessun parametro è stato aggiustato sulle tre energie.

| Molecola | QUEST | ROKS originale | Shell senza offset | Risposta completa |
|---|---:|---:|---:|---:|
| acetaldehyde | 4.304 | 4.2848 | 4.4374 | 4.5151 |
| acetone | 4.468 | 4.4681 | 4.5473 | 4.5025 |
| formamide | 5.644 | 5.5323 | 5.5879 | 5.8807 |

MAE sui tre n→π*: originale 0.0437, shell 0.0896, completo 0.1608 eV. Criterio predefinito superato: False. Tutti gli stati sono qualificati, ma la risposta completa peggiora la media rispetto a entrambi i controlli. Non può essere promossa come soluzione universale.

## Controlli conclusivi

La raffinazione della verifica Hessiana-vettore per piridazina π→π* da 2×10⁻⁴ a 10⁻⁴ e 5×10⁻⁵ bohr riduce il residuo da 3,22×10⁻⁵ a 8,05×10⁻⁶ e 2,01×10⁻⁶ Ha/bohr², coerentemente con il troncamento quadratico delle differenze centrali.

La diagnosi non lineare coerente con lo stesso λ cambia le energie verticali del modello completo, a orbitali mantenuti fissi, di +0,0242 eV in HNO e −0,0451/+0,0024 eV in uracile n→π*/π→π*. Non sono nuovi stati stazionari né correzioni da sommare alle previsioni: mostrano che i termini superiori da soli non spiegano il grande errore π→π*.

Anche le due maschere di sviluppo che conservano i blocchi di spin GFN2 sullo stesso atomo e aggiungono soltanto la risposta interatomica sono archiviate in nonlocal_screen_summary.json. Sono prove sui dieci vecchi stati di sviluppo, separate dai due test indipendenti; non giustificano una correzione universale.


## Decisione

Il candidato senza offset è distribuibile come modalità sperimentale e migliora l’accuratezza media sull’insieme indipendente più ampio. Non diventa il metodo predefinito: restano errori n→π* grandi e un bias sistematico. I modelli tensoriali locali e la sola correzione scalare della Fock non risolvono il problema; le prove negative sono conservate. Nessuna correzione universale n→π* è rivendicata.

## Riferimenti e tracciabilità

La forma standard a popolazioni di guscio e la derivazione delle costanti di spin sono discusse da [Neugebauer et al., JCC 2023](https://doi.org/10.1002/jcc.27185) e [Moradi et al., JCC 2024](https://doi.org/10.1002/jcc.27482). I nuovi termini e i loro esiti sono esperimenti di questa implementazione, non risultati rivendicati da quei lavori.

Tutti i calcoli elettronici e numerici sono stati eseguiti su oracle.sns.it, dopo verifica del nome host. Il laptop ha svolto orchestrazione, editing e verifiche di file. Source e checkpoint precedenti, colonne completate della risposta e riferimenti SCF DFT sono conservati e riutilizzati. provenance.json registra gli hash di input, script, modelli, risultati e ambiente. tests_final.log documenta 47 test passati; i controlli molecolari e delle derivate sono salvati separatamente.
