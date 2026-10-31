# ROKS 50/50 con coerenza relativa degli stati

La superficie composita conserva esattamente le energie verticali del 50/50 indipendente e mantiene durante il rilassamento gli overlap relativi dei due orbitali aperti. Le due CSF restano singoletti puri; i vincoli sono ricavati dalle funzioni d’onda assegnate, senza utilizzare le energie di riferimento. Il peso resta 1/2, λ=0,9 scala una sola volta lo scambio della risposta e non i gap. Non si aggiunge alcun offset.

Il modello è una superficie composita vincolata, non il ROKS non vincolato di un singolo funzionale DFT. Gli orbitali h/p mantengono le etichette della promozione assegnata all’ancora. Gli angoli relativi sono costanti; le orientazioni assolute e tutti gli orbitali spettatori possono rilassare. I casi di overlap unitario sono eliminati analiticamente dal sistema KKT, evitando Jacobiane singolari.

## Verifica quantitativa

- 19 stati / 18 molecole: MAE 0.1825 eV; 18 n→π*: MAE 0.1757 eV.
- Errore massimo residuo: 0.5349 eV. L’universalità della correzione non è dimostrata.
- Cinque minimi molecolari con tutti i modi reali, gap positivi e controlli indipendenti delle derivate.
- 86 test remoti passati; avvio automatico da due checkpoint qualificati, tracking APOC e restart atomico.
- Esempio pubblico verificato sia al primo avvio sia da restart: differenza energetica 1.78e-15 Ha; 561 file dei risultati trasferiti con checksum corrispondenti.

## Quattro molecole prospettiche

Questi sistemi non erano usati per sviluppare i due modelli non locali o il vincolo relativo. Le precedenti predizioni native/shell esistevano già. Il protocollo è stato fissato prima delle nuove predizioni; non si presenta il campione come una banca dati mai esaminata.

| Molecola n→π* | Riferimento | ROKS 50/50 | TDA completa su GFN2 | TDA spettrale su GFN2 |
|---|---:|---:|---:|---:|
| cyanoformaldehyde | 3.817 | 3.989 | 5.538 | 5.276 |
| cyclopropenone | 4.274 | 4.482 | 5.986 | 5.027 |
| imidazole | 6.715 | 6.180 | 6.847 | 7.703 |
| pyridazine | 3.830 | 3.530 | 4.382 | 4.924 |

MAE sui quattro sistemi: **0.304 eV** per ROKS; **1.029 / 1.073 eV** per le due TDA. Tutti gli stati passano i controlli della transizione DFT proiettata. La TDA usa gli orbitali fondamentali GFN2 corretti del rispettivo modello, con lo stesso kernel; nessun orbitale DFT entra nell’equazione TDA. La media delle due TDA è solo un controllo composito, non la risposta lineare unica del sistema KKT ROKS.

## Minimi e Hessiane analitiche

| Molecola | Gradiente massimo (Ha/bohr) | Frequenza minima (cm⁻¹) | Errore H·v a 10⁻⁴ bohr (Ha/bohr²) |
|---|---:|---:|---:|
| acetaldehyde | 2.23e-05 | 186.89 | 1.90e-09 |
| acetone | 2.51e-05 | 148.28 | 3.06e-10 |
| formamide | 5.49e-06 | 324.38 | 2.02e-09 |
| maleimide | 1.88e-05 | 179.17 | 1.96e-09 |
| nitroxyl | 9.14e-06 | 1261.41 | 3.33e-09 |

Le Hessiane comprendono la risposta KKT degli orbitali, i moltiplicatori dei vincoli, il rilassamento dei due modelli e tutte le derivate delle tre superfici S0 nella sostituzione `E0_native + mean(E_ROKS_i − E0_i)`. Non sono medie di Hessiane rilassate indipendentemente. I controlli scalare/gradiente e Hessiana-vettore usano due passi, 2×10⁻⁴ e 10⁻⁴ bohr. I singoletti sono CSF con orbitali spaziali comuni ai due determinanti di ciascuna componente; non si attribuisce una proiezione esatta a un determinante broken-symmetry ottimizzato separatamente.

È stato inoltre verificato indipendentemente ogni elemento della Hessiana differenziando tutte le componenti del gradiente:

| Sistema | Colonne | max errore Hessiana (Ha/bohr²) | max errore gradiente (Ha/bohr) |
|---|---:|---:|---:|
| formamide | 18 | 4.67e-09 | 2.26e-09 |
| nitroxyl | 9 | 1.15e-08 | 3.64e-09 |

## Tempi sul medesimo host

Calibrazione e stati S0 sono riutilizzati. Entrambi i metodi ricostruiscono i propri contesti nativi. Sono confrontati due spettri TDA completi con una singola energia ROKS già convergente, con due thread e tre ripetizioni; gradienti esclusi. Il costo per ottenere i due seed ROKS è escluso, quindi questi tempi favoriscono ROKS. Ogni matrice TDA ricostruita è confrontata con il checkpoint precedente per controllare kernel e orbitali.

| Molecola | Due TDA, tutti i root (s) | ROKS, uno stato caldo (s) |
|---|---:|---:|
| acetaldehyde | 0.0340 | 0.2983 |
| acetone | 0.0992 | 0.8690 |
| formamide | 0.0285 | 0.2470 |
| glyoxal | 0.0457 | 0.4191 |
| nitroxyl | 0.0121 | 0.0798 |

In questa implementazione ROKS non è più veloce per sole energie su queste piccole molecole. Il vantaggio verificato riguarda l’accuratezza media e l’accesso a una superficie con gradienti/Hessiane analitiche mediante energia, Fock, polarizzazione di spin e derivate del programma fondamentale GFN2, completate dal solver orbitale e dalla risposta KKT.

## Integrazione e riproducibilità

LINK seleziona `native_roks_composite_orbital_coupling="relative_active_pair"`, `native_roks_composite_following="character_newton_index"`, `native_roks_composite_reference_policy="geometry_only"` e `native_roks_composite_ground_surface="native_plus_gaps"`. La prima esecuzione legge automaticamente i due `native_roks_composite_seed_paths`, con checksum e validazione del modello/promozione/indice. I restart successivi leggono la generazione accettata completa, comprendente vincoli, moltiplicatori, due CSF e tre riferimenti S0. Il metodo resta esplicitamente selezionabile e non modifica il default.

Tutti i lavori elettronici, le ottimizzazioni e i benchmark sono stati eseguiti su **oracle.sns.it**, previa verifica dell’host. Il MacBook ha soltanto orchestrato i lavori, controllato i sorgenti e trasferito/letto i risultati. I checkpoint originali e le generazioni accettate restano disponibili sul server. Gli archivi locali selettivi omettono solo cache di calibrazione ricostruibili come `projected_eri.npz` e `quadrature_progress.npz`.

Risultati numerici completi: [SUMMARY.json](SUMMARY.json). Sorgenti/protocollo: [PROTOCOL.json](PROTOCOL.json). Log e checkpoint verificabili: `remote/`. Il confronto storico a 15 stati è una regressione esposta, non una validazione cieca. La promozione a correzione universale richiede ulteriore validazione di accuratezza, in particolare oltre il sottospazio n→π* a singola eccitazione.
