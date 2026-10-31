# HNO: correzione del riferimento SCC per ROKS 50/50

**Corretto il difetto che bloccava la ricertificazione Newton. HNO avanza,
ma non raggiunge un minimo certificato.** Tutti i calcoli sono stati eseguiti
su `oracle.sns.it`, con identità verificata prima dei lanci.

## Diagnosi corretta

Il confronto alla **stessa densità AO** mostra accordo fra Fock nativo e risposta
entro circa **1,1×10⁻¹⁶ Ha**. Cariche, dipoli e quadrupoli coincidono; i contributi
anisotropi e D4 concordano alla precisione macchina. La discrepanza della campagna
precedente confrontava il Fock esportato con un eigensistema/densità non ancora
ricertificati: non dimostrava un diverso Hamiltoniano nativo/risposta.

Per HNO il gap iniziale è appena sotto 0,01 Ha. Il controllo del riferimento
riavviava l'annealing termico dopo ogni correzione Newton, distruggendo il
miglioramento della densità e ricreando lo stesso residuo. La ricertificazione
ora usa la temperatura richiesta, conservando l'annealing ordinario per il
primo calcolo indipendente.

## Modifiche mantenute

1. Controllo `temperature_homotopy` per singola valutazione, incluso nell'identità
   della cache. Un riferimento già preparato viene ricertificato alla temperatura
   richiesta. Nessuno stato globale viene temporaneamente modificato.
2. Energie di guess S0 ordinate dopo gli spostamenti Newton. Le etichette delle
   CSF ROKS restano gestite separatamente dal tracking.
3. Corretto il ramo di uscita di un SCC diretto non convergente: risolve il
   problema generalizzato con l'overlap reale, anziché usare la matrice identità
   segnaposto dell'ortogonalizzatore. Un fallimento SCF resta un fallimento SCF,
   senza generare un falso errore di ortogonalità.
4. Al termine di DeltaSCF/ROKS, LINK non esegue un ulteriore calcolo S0 per
   scrivere descrittori SCC/Mayer che non descrivono la CSF ottimizzata. Quel
   calcolo ausiliario aveva interrotto la scrittura del risultato del secondo
   segmento; i checkpoint elettronici erano comunque già conservati.

Le soglie restano **10⁻¹⁰ Ha sul Fock** e **10⁻¹² Ha sul residuo canonico**.
Risposte congelate, λ=0,9, pesi 50/50, scaling dei gap e offset nullo non cambiano.

## Verifiche

**64 test superati su Oracle**, inclusi i 57 precedenti e regressioni per:

- riproduzione del difetto con annealing riattivato e chiusura con la correzione;
- soglie SCF di energia 10⁻⁶ e 10⁻¹⁰ Ha;
- distinzione della cache fra le due politiche termiche;
- guess energetici che cambiano ordine;
- segnalazione corretta di un SCC ripartito ma non convergente;
- assenza del calcolo S0 ausiliario alla conclusione ROKS.

Nel punto HNO conservato dalla campagna precedente, il controllo indipendente
Hessiana-vettore ora passa anche al passo che prima veniva respinto:

| Passo della differenza centrale | Max scarto dal prodotto analitico |
|---|---:|
| 2×10⁻⁴ bohr | 4,43×10⁻⁹ |
| 10⁻⁴ bohr | 1,11×10⁻⁹ |

I residui canonici dei quattro punti spostati sono fra 2,8×10⁻¹⁵ e 6,5×10⁻¹⁵ Ha.
Il calo di quattro volte dimezzando il passo è coerente con l'errore quadratico
della differenza centrale. Non è una validazione dell'accuratezza fisica della PES.

## Continuazione e limite residuo

Sono stati riutilizzati l'ultimo checkpoint accettato HNO, entrambe le componenti,
l'ancoraggio iniziale e la Hessiana dell'ottimizzatore. I segmenti e i fallimenti
intermedi sono conservati separatamente.

Il gradiente massimo scende da **0,00591036** a **0,00487298 Ha/bohr**; il criterio
richiede 0,00003. L'ultimo punto mantiene overlap fra componenti **0,9767**,
carattere attivo **0,9837 / 0,9702** e indici orbitali **1 / 0**. Il vecchio limite
di overlap completo iniziale è superato senza perdere il controllo attivo.

Il gap del riferimento S0 scende però a **1,03×10⁻⁵ Ha**. I punti successivi
presentano fallimenti SCC o mancata chiusura del riferimento entro i controlli
previsti. La Hessiana analitica dell'ultimo punto accettato è calcolabile e i
blocchi di risposta ROKS non sono singolari (minimi moduli delle curvature
0,00237 e 0,00368 Ha). Le differenze dei gradienti vicino a quell'endpoint non
superano invece il riferimento S0: non si certificano né Hessiana numerica di
controllo né frequenze di equilibrio in quel punto non stazionario.

Sono state provate una ricerca del passo Newton sul residuo nonlineare e una
sequenza Newton interna prima della ricertificazione nativa. Non hanno reso
robusta questa regione e **non sono incluse nella versione finale**. I relativi
protocolli, sorgenti sperimentali e risultati negativi restano archiviati.

Il difetto numerico iniziale è risolto e verificato; resta aperto il trattamento
del riferimento S0 quasi degenere. Questi risultati non promuovono il 50/50 per
geometrie/frequenze e non risolvono l'errore universale n→π*. Il passo successivo
richiede un trattamento robusto della quasi degenerazione, valutando anche se
l'ottimizzazione ROKS possa evitare la dipendenza da un SCC S0 stazionario a ogni
geometria, senza cambiare il funzionale eccitato o fabbricare un riferimento.

`summary.json` e `derivative_check.json` raccolgono i dati principali. Tutti i
checkpoint restano su Oracle in `/home/enzo/MOUSE-hno-fock-20261030`.


## Diagnosi dei due altri arresti, senza nuovi fit

Sono stati riletti su Oracle i checkpoint finali acetaldeide e maleimmide della
campagna precedente; non sono state rilanciate le ottimizzazioni. Il confronto
tra Hamiltoniani è qui una diagnosi esplicita, non un'autorizzazione a cambiare
la soglia durante un calcolo.

| Molecola | Overlap CSF fra componenti | Overlap coppia attiva fra componenti | Modulo determinante degli orbitali chiusi |
|---|---:|---:|---:|
| Acetaldeide | 0,9116 | 0,9674 | 0,9515 |
| Maleimmide | 0,7001 | **0,9026** | 0,8277 |

Nella maleimmide il limite della CSF completa include quindi una perdita di
sovrapposizione degli orbitali chiusi; non equivale a una perdita della stessa
entità nella coppia attiva. Anche il carattere locale rimane prevalentemente
compatibile con n→π*: la frazione fuori piano della popolazione p carbonilica
è 0,075–0,119 per la buca e 0,824–0,877 per la particella. Sarebbe ragionevole
validare un criterio intercomponente basato sulla coppia attiva, mantenendo
continuità CSF locale e controlli per ciascuna superficie. Questa campagna non
lo introduce retroattivamente e non certifica un minimo maleimmide.

Per acetaldeide la frazione fuori piano passa da circa 0 a 0,263–0,284 per la
buca e da 1 a 0,550–0,584 per la particella. È un segnale di mescolamento
significativo durante il rilassamento, coerente con il calo dell'overlap attivo
iniziale. Abbassare soltanto la soglia non risolve questa trasformazione.

Questi numeri sono descrittori nella base AO di Löwdin e nel piano carbonilico
locale, non probabilità spettroscopiche esatte né proiezioni di spin. La
risposta congelata va verificata lungo il percorso prima di adottare una
risposta DFT aggiornata con la geometria: quest'ultima richiederebbe le proprie
derivate per conservare gradienti e Hessiane coerenti.
