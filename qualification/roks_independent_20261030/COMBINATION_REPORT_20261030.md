# Combinazione dei due modelli ROKS senza offset

È stata verificata la proposta di combinare risposta completa e profilo spettrale. La media fissa delle energie riduce il MAE complessivo sui nove stati appaiati, ma non è equivalente a ottimizzare la media dei funzionali e non migliora ogni molecola. Nessun peso è stato adattato alle TBEs.

| Metodo | Stati | MAE / eV | MAE n→π* / eV (8 stati) | Errore massimo / eV |
|---|---:|---:|---:|---:|
| Risposta completa | 9 | 0.2762 | 0.2271 | 0.6992 |
| Profilo spettrale | 9 | 0.1935 | 0.2102 | 0.2944 |
| Media delle due energie, 50/50 | 9 | 0.1681 | 0.1510 | 0.4019 |
| Funzionale medio, riottimizzato 50/50 | 9 | 0.1957 | 0.1915 | 0.4830 |

## Risultati appaiati

| Molecola / stato | QUEST | Risposta completa | Spettrale | Media energie | Funzionale medio riottimizzato |
|---|---:|---:|---:|---:|---:|
| nitroxyl n_pi | 1.743 | 1.4558 | 1.5137 | 1.4848 | 1.5173 |
| uracil n_pi | 4.935 | 4.9954 | 4.7746 | 4.8850 | 4.4520 |
| uracil pi_pi | 5.381 | 6.0503 | 5.3210 | 5.6856 | 5.6097 |
| acetaldehyde n_pi | 4.304 | 4.5151 | 4.0393 | 4.2772 | 4.3309 |
| acetone n_pi | 4.468 | 4.5025 | 4.1736 | 4.3381 | 4.3873 |
| formamide n_pi | 5.644 | 5.8807 | 5.4152 | 5.6479 | 5.6823 |
| glyoxal n_pi | 2.877 | 2.8832 | 2.6210 | 2.7521 | 2.8032 |
| pyrazine n_pi | 4.144 | 3.8629 | 4.0002 | 3.9315 | 3.9333 |
| tetrazine n_pi | 2.462 | 1.7628 | 2.3574 | 2.0601 | 2.0689 |

| Gruppo | N | Completo MAE | Spettrale MAE | Media energie MAE | Funzionale medio MAE |
|---|---:|---:|---:|---:|---:|
| development | 3 | 0.3390 | 0.1499 | 0.2043 | 0.3125 |
| B_exposed | 3 | 0.1608 | 0.2626 | 0.0535 | 0.0486 |
| C_exposed | 3 | 0.3289 | 0.1681 | 0.2464 | 0.2259 |

Il gruppo C era indipendente quando è stato applicato il profilo spettrale congelato: MAE 0,1681 contro 0,3289 eV del completo e 1,1218 eV del ROKS nativo. I dati C erano già esposti quando sono state definite queste combinazioni: per esse non è una validazione cieca. Le combinazioni richiedono molecole nuove prima di essere promosse.

## Due combinazioni diverse

**Media delle energie:** Ω̄=½Ω_completo+½Ω_spettrale. Le due CSF vengono ottimizzate separatamente e devono rappresentare lo stesso stato. È un modello composito a peso fisso, non la soluzione ROKS di un singolo funzionale medio. Non si aggiunge alcuna costante specifica di stato. Migliora la media ma l’errore massimo sale a 0,4019 eV (tetrazina), contro 0,2944 eV del solo spettrale.

**Media dei funzionali:** Ē[Pα,Pβ]=½E_completo[Pα,Pβ]+½E_spettrale[Pα,Pβ], seguita da ottimizzazione separata di fondamentale ed eccitato. I due modelli condividono esattamente risposta di carica/spin e popolazione di riferimento, quindi si media soltanto il termine lineare della Fock: il kernel non viene raddoppiato. λ=0,9 resta applicato una sola volta allo scambio della risposta; non scala i gap. Il risultato non coincide con la media delle energie, per il rilassamento orbitale non lineare e per la struttura dei saddle.

Il caso più evidente è uracile n→π*: la media delle energie è 4,8850 eV, mentre il funzionale medio riottimizzato dà 4,4520 eV. La stessa soluzione è stata raggiunta sia dalla partenza canonica sia da una continuazione indipendente; il confronto non sceglie la soluzione con errore più piccolo.

È stata provata anche la combinazione senza peso libero che conserva la diagonale spettrale e recupera gli elementi non diagonali della Fock completa. Peggiora già i due n→π* di sviluppo: HNO 1,2032 contro 1,743 eV, uracile 4,2797 contro 4,935 eV. Questo ramo è stato scartato dopo i due fallimenti, conservando checkpoint e tentativi π→π* non convergenti. Non gli viene attribuita una statistica sui nove stati.

## TDA e derivate

La TDA è stata calcolata per il funzionale medio sul suo stato fondamentale GFN2, con il medesimo kernel. Le TDA dei due componenti sono anch’esse su orbitali GFN2. La media a posteriori delle loro energie TDA, quando entrambe le assegnazioni passano il gate, è solo un controllo numerico composito; non viene presentata come TDA di un singolo funzionale ROKS.

| Molecola / stato | TDA completo | TDA spettrale | Media dei valori TDA | TDA funzionale medio |
|---|---:|---:|---:|---:|
| nitroxyl n_pi | 3.4498 | 3.4171 | 3.4334 | 3.4683 |
| uracil n_pi | 5.8799 | 5.7892 | 5.8345 | 5.7835 |
| uracil pi_pi | 7.0426 | non qualificata | non qualificata | non qualificata |
| acetaldehyde n_pi | 6.2172 | 5.6910 | 5.9541 | 6.0094 |
| acetone n_pi | 6.1423 | 5.7648 | 5.9535 | 6.0045 |
| formamide n_pi | 7.3583 | 6.9456 | 7.1520 | 7.1867 |
| glyoxal n_pi | 4.1455 | 3.8521 | 3.9988 | 4.0552 |
| pyrazine n_pi | 5.1174 | 5.2081 | 5.1628 | 5.1661 |
| tetrazine n_pi | 2.9963 | 3.5947 | 3.2955 | 3.3029 |

Con pesi costanti, le derivate di una superficie composita sarebbero le medie delle derivate dei componenti: ḡ=½(g₁+g₂), H̄=½(H₁+H₂). Qui i due nuovi modelli non locali restano solo verticali: queste Hessiane nucleari dei componenti non sono ancora implementate/validate, quindi non vengono fornite come se esistessero. Le Hessiane analitiche del candidato shell precedente restano quelle già documentate e non si trasferiscono automaticamente a questi modelli.

## Scelta del prossimo candidato

La media delle energie con peso fissato 1/2 è il candidato composito più promettente per il MAE n→π* in questo confronto. Il profilo spettrale mantiene il miglior errore massimo. Il funzionale medio non offre un vantaggio netto sullo spettrale e il recupero dei soli accoppiamenti non diagonali è scartato. Nessuna combinazione è dichiarata una correzione universale, né sostituisce il metodo predefinito.

Tutti i nuovi lavori elettronici sono stati eseguiti su oracle.sns.it verificato. Il solver del riferimento TDA della pirazina è stato ripreso dal checkpoint dello stesso autostato: 1152 prodotti matrice-vettore completati conservati, SCF non ripetuto, residuo finale 1,22×10⁻¹⁰ senza cambiare root, operatore o tolleranza richiesta. Protocollo delle combinazioni e provenienza sono in results/combined_response/protocol.json e results/combined_provenance.json.
