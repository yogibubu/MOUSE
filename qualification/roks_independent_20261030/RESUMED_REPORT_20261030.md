# Ripresa ROKS: diagnosi della risposta e nuova prova indipendente

Esperimenti del 30 ottobre 2026, interamente su oracle.sns.it. Si riutilizzano SCF DFT, modelli di risposta e checkpoint precedenti. Nessun offset di stato e nessun adattamento di λ alle energie eccitate. Il rapporto precedente REPORT.md e la sua provenienza restano separati.

## Separazione dei termini

Sette ablation del medesimo funzionale, oltre al modello completo già calcolato, su HNO, due stati dell’uracile e i tre carbonili già esposti. Le tre componenti sono il termine lineare della Fock, la risposta di carica e la sostituzione della risposta di spin. Lo spin GFN2 nativo viene ricostruito esattamente nella base di coppie AO e controllato in energia e Fock prima degli esperimenti. Non si sceglie un profilo sulla base dei nuovi risultati.

| Profilo | Stati | MAE / eV | MAE n→π* / eV |
|---|---:|---:|---:|
| native_control | 6 | 0.3110 | 0.1522 |
| fock_only | 6 | 2.9625 | 3.2016 |
| charge_only | 6 | 1.2684 | 1.1604 |
| spin_only | 6 | 2.1887 | 2.3033 |
| fock_charge | 6 | 2.1025 | 2.3083 |
| fock_spin | 6 | 1.0108 | 0.9471 |
| charge_spin | 6 | 3.1587 | 3.3259 |
| full | 6 | 0.2499 | 0.1660 |

Le 48 energie vengono ricostruite dalla somma dei contributi con errore massimo 3.366e-13 eV. La decomposizione non modifica le energie e non rappresenta una correzione da sommare alle previsioni.

| Stato, modello completo | GFN2 sugli orbitali del modello | Fock | Carica | Sostituzione spin | Totale |
|---|---:|---:|---:|---:|---:|
| nitroxyl n_pi | 1.8045 | +3.3376 | -1.1933 | -2.4930 | 1.4558 |
| uracil n_pi | 4.5991 | +2.9977 | -0.7699 | -1.8315 | 4.9954 |
| uracil pi_pi | 4.3828 | +2.9351 | -0.6925 | -0.5751 | 6.0503 |
| acetaldehyde n_pi | 4.5012 | +3.4137 | -0.9269 | -2.4728 | 4.5151 |
| acetone n_pi | 4.6119 | +3.1172 | -0.8186 | -2.4080 | 4.5025 |
| formamide n_pi | 5.7495 | +3.5385 | -0.9573 | -2.4501 | 5.8807 |

I termini trasferiti si compensano per diversi eV. La sola correttezza delle derivate non garantisce accuratezza delle eccitazioni; trasferire soltanto carica o spin può produrre energie negative rispetto al fondamentale chiuso. Questi risultati numericamente stazionari sono conservati nella tabella, senza rimuoverli dalle statistiche delle ablation.

## Stabilità elettronica

Verificate 30 curvature elettroniche reali singoletto/tripletto con differenze centrali di energia: errore massimo 3.030e-06 Ha. Non sono Hessiane nucleari. In HNO il modello completo ha un modo tripletto negativo (autovalore di H/4 = −0,020353 Ha). La variante spettrale sotto descritta conserva un modo negativo (−0,013997 Ha).

Il riferimento PBE0 completo, riletto dal checkpoint senza ripetere SCF, è anch’esso instabile: H/4 = -0.01179987 Ha, residuo Davidson 4.767e-08. Questo controllo usa PBE0 a=0,25, mentre il modello compatto usa λ=0,9 nella risposta. L’instabilità non viene quindi attribuita soltanto alla proiezione GFN2 e non viene rimossa con un’aggiunta empirica.

## Trasporto delle energie orbitali DFT

Il trasporto ortogonale dell’intero spettro viene rifiutato prima delle previsioni: gli spazi virtuali selezionati hanno valori singolari minimi 0,4925 in HNO e 0,0443 nell’uracile, sotto la soglia fissata 0,5. Il pivoting QR non risolve il problema dell’uracile (0,1687). Le singole coppie n/π* e π/π* di interesse hanno invece mapping affidabile.

È stato quindi costruito un unico profilo distinto, qualified-spectral-full-response: termine lineare diagonale nella base canonica GFN2, livelli DFT trasferiti solo con overlap ≥0,7 e margine ≥0,05, energie non qualificate mantenute native nella stessa gauge del centroide occupato. Il termine è parte del funzionale variazionale e cambia la Fock; non è una costante aggiunta all’energia di uno stato. La risposta completa di carica/spin resta identica. λ=0,9 agisce una sola volta sullo scambio della risposta; i gap DFT non vengono moltiplicati per λ.

| Dati già esposti | QUEST | Profilo spettrale | Errore |
|---|---:|---:|---:|
| nitroxyl n_pi | 1.743 | 1.5137 | -0.2293 |
| uracil n_pi | 4.935 | 4.7746 | -0.1604 |
| uracil pi_pi | 5.381 | 5.3210 | -0.0600 |
| acetaldehyde n_pi | 4.304 | 4.0393 | -0.2647 |
| acetone n_pi | 4.468 | 4.1736 | -0.2944 |
| formamide n_pi | 5.644 | 5.4152 | -0.2288 |

Tre stati di sviluppo: MAE 0.1499 eV.

Tre carbonili già esposti: MAE 0.2626 eV.

Tutti i cinque n→π* esposti: MAE 0.2355 eV.

Il miglioramento dell’uracile π→π* si accompagna a un peggioramento sistematico dei carbonili. Questi ultimi non sono una nuova prova cieca: i loro risultati precedenti erano già noti. Il profilo resta invariato nel successivo test C.

## Tre molecole indipendenti: insieme C

Protocollo congelato prima delle previsioni, SHA256 ba61d18eebee68e7dee9ca6d0d69b618ad9d87f50ca305f987df6c7534ff583f. Gliossale, pirazina e tetrazina non hanno selezionato il modello. Primo stato n→π* singoletto valenza sicuro QUEST; root DFT TDA 1 fissata senza sostituzioni, proiezione e identità ≥0,7, residuo orbitale ≤10⁻⁹. Tutti e tre gli stati sono richiesti dal criterio di successo.

| Molecola | QUEST | ROKS nativo | Risposta completa precedente | Profilo spettrale | TDA stesso funzionale/GFN2 |
|---|---:|---:|---:|---:|---:|
| glyoxal | 2.877 | 2.1054 | 2.8832 | 2.6210 | 3.8521 |
| pyrazine | 4.144 | 2.7606 | 3.8629 | 4.0002 | 5.2081 |
| tetrazine | 2.462 | 1.2517 | 1.7628 | 2.3574 | 3.5947 |

native: N=3, MAE=1.1217578543457962, massimo assoluto=1.3833603002638775 eV.

raw_full: N=3, MAE=0.32885331161255277, massimo assoluto=0.6992216386000256 eV.

spectral: N=3, MAE=0.16814290732284634, massimo assoluto=0.2560021198409954 eV.

Criterio indipendente superato: **True**. Il profilo non viene modificato dopo aver osservato C.

La TDA usa gli orbitali GFN2 ottenuti con lo stesso funzionale del profilo spettrale, con la stessa risposta di carica. Gli orbitali DFT servono alla calibrazione e all’assegnazione di riferimento, non sostituiscono la base delle transizioni TDA. Tempi di soluzione ROKS e TDA, e costo separato della preparazione DFT, sono nei JSON/NPZ. Non si rivendica un vantaggio generale di tempo.

## Continuità e limiti delle derivate

Le partenze canoniche possono convergere a saddle di diversa identità: questi tentativi e i relativi checkpoint non vengono eliminati. La continuazione della Fock completa verso quella spettrale registra overlap tra CSF consecutive e rispetto alla promozione originale. Passaggi intermedi t<1 sono diagnostiche numeriche, non nuovi parametri selezionabili o previsioni del profilo finale. I tentativi che perdono l’identità originale restano esclusi.

Questo prototipo non locale è **solo verticale**. Non gli vengono attribuite le Hessiane nucleari analitiche già validate per il candidato shell. I 47 test precedenti e le otto superfici con Hessiana analitica rimangono quelli documentati in REPORT.md; qui sono aggiunti controlli elettronici indipendenti, non una nuova implementazione di derivate nucleari.

## Esito

La correzione universale n→π* resta irrisolta: il profilo spettrale riduce l’errore su alcuni stati ma peggiora i carbonili già esposti. I nuovi risultati distinguono errore del funzionale, instabilità del riferimento e perdita di identità dello stato. Non si introducono offset, λ per molecola o correzioni scelte sulle TBEs. Nessun metodo predefinito è cambiato.

Dati molecolari: [QUESTDB](https://github.com/pfloos/QUESTDB), commit 460482bcc6231dab41fe32188dc43959a7c2bdaa. Metadati, geometrie, licenza, checkpoint, script eseguiti e risultati sono conservati; resumed_provenance.json registra gli hash della ripresa separatamente da provenance.json della campagna precedente.
