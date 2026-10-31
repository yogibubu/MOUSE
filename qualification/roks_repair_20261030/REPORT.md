# ROKS, risposta DFT e validazione molecolare — 30 ottobre 2026

Il codice e le derivate sono verificati. La correzione economica selezionata sui dati di sviluppo **fallisce** sulle nuove molecole e non diventa il metodo predefinito. L’ancoraggio a energie DFT completamente rilassate è disponibile come modello locale esplicito; non costituisce una nuova validazione cieca.

## Accuratezza e generalizzazione

La scelta BH&HLYP (energia congelata) + risposta PBE0 di carica/spin proiettata su GFN2 è stata fissata prima dei nuovi calcoli. λ=0,9 moltiplica una sola volta lo scambio HF nella risposta di riferimento; non moltiplica gap o energie ROKS. Nessun λ dipendente dall’estensione è stato promosso.

| Insieme | Modello | N | MAE (eV) | Errore massimo (eV) |
|---|---|---:|---:|---:|
| development | ROKS SE originale | 10 | 0.6257 | 1.2132 |
| development | PBE0 congelato + rilassamento SE | 10 | 0.2551 | 0.4489 |
| development | Candidato BH&HLYP congelato + risposta corretta | 10 | 0.1955 | 0.5903 |
| development | ROKS SE con risposta corretta, senza ancoraggio | 10 | 0.5472 | 1.0187 |
| development | TDA nativa su orbitali GFN2 | 9 | 0.7929 | 1.5336 |
| development | TDA con risposta corretta, su orbitali GFN2 | 9 | 0.7880 | 1.5336 |
| development | Ohno J/K, profilo PBE0 λ=0,9 su orbitali GFN2 | 10 | 2.6516 | 3.7206 |
| development | ROKS BH&HLYP completamente rilassato (controllo successivo) | 10 | 0.3044 | 0.7400 |
| holdout | ROKS SE originale | 3 | 0.1084 | 0.1875 |
| holdout | PBE0 congelato + rilassamento SE | 3 | 0.7589 | 1.0309 |
| holdout | Candidato BH&HLYP congelato + risposta corretta | 3 | 0.9188 | 1.1014 |
| holdout | ROKS SE con risposta corretta, senza ancoraggio | 3 | 0.0960 | 0.1395 |
| holdout | TDA nativa su orbitali GFN2 | 3 | 0.4415 | 0.5311 |
| holdout | TDA con risposta corretta, su orbitali GFN2 | 3 | 0.4415 | 0.5311 |
| holdout | Ohno J/K, profilo PBE0 λ=0,9 su orbitali GFN2 | 3 | 2.5019 | 2.8085 |
| holdout | ROKS BH&HLYP completamente rilassato (controllo successivo) | 3 | 0.2017 | 0.4010 |

| Insieme appaiato | Modello | N comune | MAE (eV) |
|---|---|---:|---:|
| development | native_roks_ev | 9 | 0.5725 |
| development | matched_roks_ev | 9 | 0.5065 |
| development | native_tda_ev | 9 | 0.7929 |
| development | matched_tda_ev | 9 | 0.7880 |
| holdout | native_roks_ev | 3 | 0.1084 |
| holdout | matched_roks_ev | 3 | 0.0960 |
| holdout | native_tda_ev | 3 | 0.4415 |
| holdout | matched_tda_ev | 3 | 0.4415 |

Le statistiche TDA/ROKS su identiche righe qualificate sono conservate anche in summary.json (tda_roks_paired). Le medie con numerosità diversa non sono confronti appaiati.

Esito del gate congelato: **False**. Peggioramento individuale massimo: 0.4810 eV (limite 0,25 eV). Le medie DFT rilassate possono includere meno stati: confrontare sempre le righe comuni della tabella seguente.

Il risultato senza offset energetico è un’analisi successiva del modello, non il superamento del gate del candidato congelato: richiede un’ulteriore prova indipendente prima di promuoverlo. Il pirrolo resta escluso: il riferimento B2 ha proiezione di transizione di circa 0,48 nella base compatta GFN2, sotto la soglia fissata 0,7. Non sono state cambiate simmetrie o soglie. La citosina n→π* resta esclusa per assegnazione ambigua nel precedente protocollo. L’uracile π→π* non supera il gate di identità TDA e resta fuori dai confronti TDA/ROKS appaiati.

| Molecola | Stato | QUEST (eV) | ROKS SE | TDA GFN2 | Candidato congelato | DFT rilassato |
|---|---|---:|---:|---:|---:|---:|
| nitroxyl | n_pi | 1.743 | 1.7285 | 1.1118 | 1.5521 | 1.3495 |
| uracil | n_pi | 4.935 | 4.3194 | 3.7278 | 4.9264 | 4.7612 |
| uracil | pi_pi | 5.381 | 4.2762 | — | 5.6479 | 5.5598 |
| thymine | n_pi | 4.980 | 4.3338 | 3.8162 | 5.2008 | 4.8098 |
| thymine | pi_pi | 5.279 | 4.0658 | 4.6942 | 5.4183 | 5.4090 |
| nitrous_acid | n_pi | 3.527 | 3.0537 | 2.5483 | 3.5138 | 3.1073 |
| propynal | n_pi | 3.808 | 3.5817 | 3.2087 | 4.2862 | 3.4413 |
| pyridine | n_pi | 4.959 | 3.8929 | 3.4254 | 4.9179 | 4.8734 |
| pyridine | pi_pi | 5.133 | 4.8269 | 5.0075 | 5.7233 | 5.8730 |
| cytosine | pi_pi | 4.736 | 4.1449 | 4.4235 | 4.7417 | 5.1218 |
| ketene | pi_pi | 3.860 | 4.0475 | 3.5797 | 2.9393 | 3.6676 |
| diazomethane | pi_pi | 3.047 | 3.1600 | 2.5159 | 1.9456 | 2.6460 |
| nitroaniline | n_pi | 3.998 | 4.0227 | 3.4848 | 4.7321 | 4.0097 |

## Costo: confronto coerente

La TDA denominata nativa nella tabella usa il kernel di risposta SCC GFN2. La TDA corretta usa U ottenuto dalla risposta DFT. Il controllo separato Ohno J/K usa il profilo PBE0 con coefficiente finale di scambio 0,25×0,9=0,225 e filtro spaziale 0,5, applicato agli orbitali/gap GFN2: questi ultimi non contengono una frazione HF 0,25; è una convenzione del kernel di riferimento, non una proprietà GFN2. I risultati mostrano che trasferire questo kernel senza un bilanciamento coerente dei gap produce errori grandi; non si confonde tale controllo con una TDA dello stesso funzionale SE. Le matrici TDA usano davvero gli orbitali canonici GFN2 dello stato fondamentale. I gap DFT mappati sono un esperimento distinto, salvato nei risultati grezzi, che produce errori grandi e non viene selezionato. Nessun confronto con TDA su orbitali DFT viene presentato come equivalente.

| Nuova molecola | TDA, costruzione condivisa + diagonalizzazione (s) | ROKS risposta corretta da orbitali canonici (s) |
|---|---:|---:|
| ketene | 0.02323 | 0.29600 |
| diazomethane | 0.00994 | 0.38137 |
| nitroaniline | 2.40710 | 7.38017 |

Questi tempi escludono la preparazione DFT, che è un costo aggiuntivo del modello corretto. La TDA produce più radici per matrice; ROKS ottimizza uno stato. I tempi di sviluppo ROKS partono da checkpoint rilassati e non sono confronti di avvio a freddo. Non si osserva un vantaggio generale di velocità ΔSCF.

## Modello variazionale e derivate

Per ciascuno stato qualificato si salva un modello locale: E(R)=E_ROKS,GFN2+risposta_congelata(R)+C. W di spin, U di carica, popolazioni di riferimento e C restano fissi. La metrica AO e la densità dipendono dalla geometria. Per kind=relaxed_vertical, C riproduce a R0 l’energia di eccitazione ROKS DFT completamente rilassata. Una costante di stato non migliora le forze; queste restano quelle del modello SE corretto.

Il singoletto è una CSF S=0 a due determinanti con orbitali comuni; si ottimizza E_S=2E_misto−E_tripletto. Non si rivendica la proiezione esatta dell’energia di un determinante UKS rotto. La ricerca GMF ammette curvature elettroniche negative; l’indice è verificato contro gli autovalori dell’Hessiana orbitale SE. Per i riferimenti DFT si verificano residuo orbitale ≤10⁻⁷ e overlap CSF ≥0,7, senza certificare l’indice completo.

L’Hessiana nucleare è analitica e include il complemento di Schur della risposta orbitale, i contributi nativi GFN2, spin, carica e metrica. Non usa geometrie spostate. Le differenze finite sono controlli indipendenti: HNO, tutte le 9 colonne, errore massimo 1,55×10⁻⁶ Ha/bohr²; uracile, Hessiane complete 36×36, controlli direzionali casuali e stretching C=O, errori massimi 4,55×10⁻⁶ (nπ*) e 2,65×10⁻⁶ (ππ*) Ha/bohr². Queste sono derivate esatte del modello congelato, non Hessiane DFT e non derivate di parametri ricalibrati a ogni geometria.

APOC conserva gli overlap coerenti delle CSF; LINK usa lo stesso modello per energie, gradienti e Hessiane, con fingerprint del funzionale e checksum del file nel restart. I punti accettati aggiornano il riferimento. Test: 43 passati su Oracle; controllo molecolare reale LINK/APOC in results/link_molecular_validation.json.

## Uso e limiti

I file results/relaxed_anchors/MOLECOLA/STATO/model.npz contengono il modello completo senza pickle; central.npz il checkpoint, analytic.npz gradiente/Hessiana/curvature, backend.json la configurazione LINK. I percorsi nel backend sono quelli Oracle e vanno aggiornati insieme al file modello quando si trasferisce il lavoro. Il checksum del contenuto resta valido. Ogni validation.json dichiara lo scopo e verifica energia, risposta orbitale, traslazioni e indice.

I riferimenti DFT usano BH&HLYP/aug-cc-pVDZ cartesiana, griglia 3 e ottimizzazione diretta degli orbitali. Il congelamento iniziale degli orbitali aperti è seguito dal rilascio di tutte le rotazioni attive. Checkpoint già completati sono riutilizzati. Un riferimento che perde identità non viene sostituito da uno stato più basso. Le eventuali failure.json conservano i tentativi, anche quando una successiva ripresa riesce: result.json qualificato è l’esito finale.

Conclusione scientifica: riuso del programma di stato fondamentale e derivate analitiche sono dimostrati; una correzione semiempirica universale dell’errore n→π* non è dimostrata. Gli ancoraggi rilassati pagano un calcolo DFT per stato e non possono essere usati come prova che la sola approssimazione SE abbia raggiunto quell’accuratezza. Il metodo predefinito resta invariato.

## Lavori e idee di Levi

[Levi, Ivanov e Jónsson, Faraday Discussions 2020](https://doi.org/10.1039/D0FD00064G), [Schmerwitz, Levi e Jónsson, JCTC 2023, Generalized Mode Following](https://doi.org/10.1021/acs.jctc.3c00178) e [Freeze-and-Release Direct Optimization, JCTC 2026](https://doi.org/10.1021/acs.jctc.5c01974) completano i riferimenti metodologici. [Levi, Ivanov e Jónsson, JCTC 2020, Variational Density Functional Calculations of Excited States via Direct Optimization](https://doi.org/10.1021/acs.jctc.0c00597): ottimizzare direttamente le rotazioni orbitali verso soluzioni stazionarie eccitate. [Documentazione ufficiale GPAW](https://gpaw.readthedocs.io/documentation/do/do.html): ricerca di saddle e preparazione con orbitale buco/particella congelati prima del rilascio. Queste idee motivano la ricerca e il controllo di identità; non forniscono una parametrizzazione universale GFN2.

## Tracciabilità

Tutti i lavori elettronici e numerici sono stati eseguiti su oracle.sns.it dopo verifica del nome host. scripts, checkpoint, colonne di risposta, matrici TDA, gradienti/Hessiane e protocolli sono conservati. summary.json mantiene i valori completi; provenance.json registra ambiente e hash SHA256. Il laptop ha svolto solo orchestrazione, modifica del testo/codice e trasferimenti.
