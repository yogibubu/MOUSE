# ROKS 50/50: continuità elettronica e continuazioni molecolari

Calcoli esclusivamente su `oracle.sns.it`, identità verificata prima dei lanci.
Le risposte DFT congelate, λ=0,9 sullo scambio, i gap senza scaling aggiuntivo,
i pesi 50/50 e l'assenza di offset energetici restano invariati.
Non sono state rifatte le calibrazioni o le fasi già completate.

## Interventi nel codice

- Continuità locale: overlap completo APOC della CSF rispetto all'ultimo punto
  accettato, sempre almeno 0,7 per ciascuna componente.
- Carattere lungo il percorso: overlap della coppia attiva rispetto allo stato
  iniziale, in AO simmetricamente ortogonalizzati e nel riferimento corotante
  della risposta congelata. L'overlap della diade simmetrica è
  `abs((h0·h1)(p0·p1) + (h0·p1)(p0·h1))`; tollera uno scambio delle due
  etichette orbitali, che rappresenta la stessa CSF singoletto. Soglia 0,7.
  L'overlap completo iniziale rimane nel log, senza diventare un vincolo
  improprio sul prodotto degli orbitali spettatori lungo un grande rilassamento.
  Questa misura descrive carattere nella base AO scelta: non certifica da sola
  l'assegnazione spettroscopica n→π* né l'accuratezza della superficie.
- Per HNO, tentativo limitato all'indice orbitale adiacente osservato dal GMF,
  entro ±1 dall'indice iniziale, dallo stesso checkpoint accettato. Nessuna scelta
  del root in base all'energia o ai dati di riferimento.
- Recupero opzionale dello stato fondamentale: se il riferimento indipendente
  fallisce, un solo nuovo tentativo dal suo S0 accettato, proiettato tramite
  l'overlap AO fra geometrie. Identiche soglie di chiusura Fock e canonicità.
- Un confronto fallito fra le due componenti del 50/50 viene comunicato a LINK
  come errore recuperabile di continuità: il passo viene respinto. La soglia
  intercomponente 0,7 resta attiva.
- Checkpoint atomici per entrambe le componenti; contratti distinti per le nuove
  opzioni e migrazione esplicita dall'ultima geometria accettata. Conservati gli
  ancoraggi originali e riutilizzate le Hessiane dell'ottimizzatore.

Il comportamento preesistente rimane predefinito. Le opzioni sono documentate
nel README; nessuna opzione nuova modifica l'energia o le derivate del modello.

## Validazione e limiti

**57 test superati su Oracle.** I test remoti coprono invarianza per rotazione/traslazione e fase, rilassamento
spettatore, cambiamento della coppia attiva, scambio equivalente degli orbitali
aperti, rifiuto locale e globale, limiti del tentativo GMF, identità dei restart,
recupero S0 e gestione del disaccordo fra componenti. Sono inclusi i precedenti
controlli delle derivate e della selezione della Hessiana finale eccitata.

I protocolli sono salvati prima dei rispettivi lanci. Le estensioni di recupero
numerico sono dichiarate separatamente dal primo protocollo di tracking.
Le continuazioni partono dai checkpoint della campagna precedente: HNO dal
ramo principale; acetaldeide e maleimmide dal ramo non planare negativo.

## Esiti conclusivi

**Zero minimi certificati su tre molecole. Tutti i processi sono terminati.**
I nuovi controlli permettono di superare il precedente limite dell'overlap
completo con la geometria iniziale, ma non risolvono da soli la validità della
superficie lungo l'intero rilassamento.

| Molecola | Ultimo max gradiente, Ha/bohr | Controllo determinante | Esito |
|---|---:|---|---|
| HNO | 0,00591036 | Carattere attivo 0,9916 / 0,9837; overlap fra componenti 0,9801 | Arresto del riferimento S0; nessun minimo |
| Acetaldeide | 0,03158583 | Carattere attivo 0,7434 / **0,700064** | 80 passi di recupero; limite di carattere iniziale |
| Maleimmide | 0,03478515 | Overlap fra componenti **0,700083**; carattere attivo 0,9336 / 0,8293 | Arresto dopo 21 passi di recupero; limite intercomponente |

Per HNO, il passaggio dell'indice della componente raw da 0 a 1 conserva
continuità e carattere. Il gradiente scende da 0,0169205 a 0,0059104. La diagnosi
successiva misura una differenza di **4,587×10⁻⁹ Ha** fra il Fock esportato dal
calcolo nativo e il Fock ricostruito dalla risposta alla stessa densità.
Il tentativo dal S0 accettato non elimina questo limite. Non è stato mascherato
abbassando i requisiti di chiusura o canonicità.

Per acetaldeide, il primo segmento si arresta a 0,0125219 per un fallimento del
S0 indipendente. Il recupero proiettato risolve un punto di discesa precedentemente
respinto con residuo Fock **5,35×10⁻¹⁴ Ha**, senza modificare le soglie. Gli 80
passi successivi proseguono fino al limite del carattere attivo, con C=O
**1,47705 Å**: non è una geometria di equilibrio da confrontare come minimo con
la letteratura. Il riferimento S1 CCSDR(3) già documentato nella campagna
precedente ha C=O 1,331 Å. Il rilassamento ulteriore non costituisce quindi un
miglioramento dimostrato dell'accuratezza.

Per maleimmide, il primo segmento termina su un trial con overlap fra componenti
0,688258. La gestione corretta del trial consente una continuazione dal checkpoint
accettato, senza autorizzare quel punto. Il segmento successivo arriva a 0,700083
e si arresta. Questo diagnostica il limite del criterio fra le due soluzioni;
un overlap basso non prova, da solo, un diverso carattere spettroscopico.

### Verifica delle Hessiane analitiche

Scarto massimo fra prodotto Hessiana-vettore analitico e differenza centrale
dei gradienti, nei punti accettati indicati:

| Punto | Passo 2×10⁻⁴ bohr | Passo 10⁻⁴ bohr |
|---|---:|---:|
| HNO, termine del primo segmento | riferimento S0 respinto | 1,11×10⁻⁹ |
| Acetaldeide, termine del primo segmento | 1,02×10⁻⁹ | 2,58×10⁻¹⁰ |
| Acetaldeide, termine del recupero | 6,97×10⁻⁹ | 1,74×10⁻⁹ |
| Maleimmide, checkpoint del primo segmento | 5,67×10⁻⁷ | 1,42×10⁻⁷ |

Il calo di circa quattro volte dimezzando il passo è coerente con l'errore
quadratico della differenza centrale. I controlli riguardano le derivate della
stessa PES congelata; non ne dimostrano l'accuratezza fisica. Nessun punto è
stazionario: le curvature salvate non sono frequenze di equilibrio. La Hessiana
al termine del secondo segmento maleimmide non è stata richiesta, poiché quel
punto non supera il criterio sul gradiente.

## Conclusione operativa

L'integrazione distingue ora continuità locale, carattere iniziale, indice
orbitale e fallimenti del riferimento S0, senza cambiare le energie del 50/50.
Le derivate analitiche risultano coerenti nei nuovi controlli molecolari.
Il candidato resta sperimentale: questi risultati non autorizzano l'uso generale
per ottimizzazioni e frequenze, né una correzione universale dell'errore n→π*.
I controlli sul carattere restano attivi; non si propone di risolvere gli arresti
abbassando le soglie o rifittando λ sulle tre molecole.

Il problema numerico da affrontare per primo è la coerenza Fock nativo/risposta
in HNO. Per il modello energetico serve poi verificare il dominio di validità
della risposta congelata e la corrispondenza delle due componenti lungo i
rilassamenti, prima di una nuova validazione indipendente più ampia.

I risultati di questa campagna non costituiscono una nuova validazione cieca
dell'accuratezza delle energie verticali. Tempi di recupero: 129,55 s per gli 80
passi acetaldeide, 255,87 s per i 21 passi maleimmide, esclusa la preparazione
precedente e gli audit. Non sono confronti di velocità equivalenti con TDA.

`summary.json` raccoglie gli endpoint accettati; `provenance.json` e
`TRANSFER_VERIFICATION.json` permettono di controllare gli artefatti trasferiti.
I checkpoint completi restano anche su Oracle sotto
`/home/enzo/MOUSE-roks-following-20261030`.

## Aggiornamento della diagnosi HNO

Il [confronto successivo alla stessa densità](../hno_fock_20261030/REPORT.md)
mostra accordo nativo/risposta alla precisione macchina. La precedente ipotesi
di incoerenza fra i due Hamiltoniani è quindi superata: il residuo era legato
alla ricertificazione degli orbitali e alla riattivazione dell'annealing dopo
Newton. Il nuovo rapporto distingue la correzione verificata dal successivo
limite di quasi degenerazione del riferimento S0. Gli artefatti originali
rimangono conservati e non sono stati alterati.
