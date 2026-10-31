# Nuovo funzionale: sviluppo e verifica del 30 ottobre 2026

**Decisione: non sostituire il 50/50.** Il candidato ispirato alla correlazione di ensemble migliora il pannello esplorativo n→π*, ma non supera il controllo prospettico su tre molecole nuove. Nessun offset o parametro è stato riadattato ai risultati.

## Funzionale implementato

Il candidato usa il funzionale spin-polarizzato PBE0/GFN2 già calibrato per il tripletto ausiliario e ottimizza gli orbitali del singoletto comune con

`E_S = E_T + 2 (1 − ξ) (ab|ab)`, con `ξ = 0,32` pubblicato e fisso.

Il termine dipende dagli orbitali e dalla densità di transizione. Energia, gradiente orbitale e Hessiana elettronica includono la sua risposta durante la ricerca del punto stazionario; non è un aggiustamento energetico successivo. Il riferimento fondamentale è ricalcolato con lo stesso funzionale parentale. La CSF a due determinanti è un singoletto spin-puro. λ=0,9 resta applicato una sola volta allo scambio del kernel; il termine Hartree di ensemble non riceve un secondo scaling.

Sono stati implementati anche un kernel a separazione di raggio (HF a corto raggio 3/8, HF a lungo raggio 1, omega=0,2 bohr⁻¹) e riferimenti DFT coerenti con quel kernel. Queste sono trasposizioni di ricerca nel modello GFN2: **non equivalgono al GX24 completo**. GX24 include un funzionale nonlineare completo e correlazione di ensemble; la nostra risposta congelata e i termini superiori GFN2 sono approssimazioni ulteriori.

## Accuratezza

| Insieme | Nuovo candidato | Vecchio spettrale | Vecchio 50/50 |
|---|---:|---:|---:|
| 11 n→π* già esaminati | 0.129321 | 0.187730 | 0.156371 |
| 3 molecole nuove F | 0.155148 | 0.168521 | 0.076258 |

MAE in eV. Tutti e tre gli stati del test F soddisfano le soglie di identità e stazionarietà. Il ROKS nativo ha MAE 0,722922 eV su F.

| Molecola nuova | Riferimento | Nuovo candidato | Vecchio 50/50 |
|---|---:|---:|---:|
| cyclopentadienone | 2.951000 | 2.825257 | 2.953347 |
| maleimide | 3.810000 | 3.605430 | 3.643225 |
| pyranone | 4.016000 | 4.151131 | 4.075652 |

Sui 14 n→π* complessivi il MAE è 0.134855 eV per il candidato e 0.139204 eV per il 50/50. La differenza aggregata è piccola e non compensa il peggioramento nel nuovo test. **Il criterio F prefissato è fallito**: il candidato non batte il 50/50. I pesi del controllo rimangono 50/50.

F comprende ciclopentadienone, maleimmide e piranone, fissati prima delle predizioni del candidato. Sono nuovi rispetto allo sviluppo MOUSE; non è stata certificata l’assenza di sovrapposizione con il training originario del parametro ξ pubblicato.

## Stati mancanti e tentativi respinti

Il controllo π→π* dell’uracile non raggiunge il punto finale con la sovrapposizione minima alla transizione DFT fissata. La continuazione PBE0 si arresta a θ=0,992578, con θ=1 richiesto dal funzionale. Questo valore intermedio non viene contato come predizione. Il criterio generale che richiedeva anche quello stato resta fallito.

L’adattamento RSH con primo riferimento PBE0 mantenuto fisso produce anche un gap negativo per HNO e viene scartato. Usando invece livelli e kernel RSH coerenti il calcolo supera le verifiche matematiche, ma il ramo completo sovrastima fortemente alcuni stati e la continuazione spettrale sull’uracile non conserva lo stato fino a θ=1. Non si assegna un MAE generale ai soli successi.

Il distinto insieme E (diazete, metanimina, triazina) è stato preparato ma non usato per predizioni del candidato, poiché il suo controllo preliminare sull’uracile falliva. La triazina ha simmetria D3h A1 doppio-primo verificata con un proiettore di gruppo (peso 0,999999995), ma il seed nativo ha sovrapposizione alla transizione 0,696937, inferiore a 0,7: resta esclusa, senza sostituzioni. F è un successivo test dichiarato e separato per il candidato S1 n→π*, non un rimpiazzo nascosto della triazina.

## Risposta, tracking e costo

Le verifiche indipendenti del kernel RSH contro la risposta UKS PySCF passano per HNO e uracile, con errori di matrice fino a circa 6e−13 hartree. La risposta del nuovo termine quartico passa controlli su quattro direzioni indipendenti tramite sole energie; il limite Hartree–Fock soddisfa l’identità esatta singoletto/tripletto entro 5e−15 hartree.

È stato corretto il tracking durante il cambio di funzionale: si richiedono sovrapposizione alla CSF eccitata qualificata, continuità fra passi e proiezione sulla transizione DFT originale, tutte almeno 0,7. La sovrapposizione al determinante canonico GFN2 iniziale viene conservata come diagnostico. I protocolli precedenti e i loro arresti restano archiviati; non vengono retroattivamente dichiarati superati. I checkpoint sono stati migrati verificando che il funzionale energetico fosse identico, senza rifare i passi completati.

I controlli TDA sono calcolati sugli orbitali GFN2 del fondamentale del medesimo funzionale parentale, non su orbitali DFT nella base grande. Il termine di correlazione di ensemble del candidato è specifico dello stato: questi controlli sono TDA del parentale, non un’affermazione di identità fra TDA e il nuovo funzionale di stato.

MAE TDA parentale su F: raw_full=1.055014 eV (3 stati), old_spectral=0.934408 eV (3 stati).

Il prototipo non dimostra un vantaggio di tempo su TDA. I tempi di continuazione e di costruzione/diagonalizzazione TDA sono riportati separatamente nel riepilogo JSON; non includono il costo dei riferimenti DFT e non costituiscono un benchmark controllato. Il solver usa energia/Fock del programma fondamentale, ma questa implementazione con Hessiane orbitali dense ha un costo proprio significativo.

**L’Hessiana qui implementata è elettronica/orbitale. Gradienti nucleari e Hessiana nucleare analitica del nuovo funzionale non sono implementati.** Le precedenti Hessiane nucleari native/shell/tensor non vengono estese automaticamente a questi modelli nonlocali.

Tutto il lavoro elettronico è stato eseguito su oracle.sns.it. Sono conservati riferimenti, storie, checkpoint e fallimenti; nessun default di MOUSE è stato cambiato.

## Fonti e artefatti

[Gould, Dale, Kronik e Pittalis, PRL 134, 228001 (2025)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.228001); [formulazione e parametri di ensemble](https://arxiv.org/html/2504.00547v3); [rassegna OO-DFT di Levi e collaboratori](https://arxiv.org/html/2606.13654v1).

Dettagli riproducibili: results/edft_functional_final_summary.json, results/holdout_F_summary.json, edft_holdout_F_protocol.json e results/edft_provenance.json.
