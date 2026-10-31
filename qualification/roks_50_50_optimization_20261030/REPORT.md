# Prova molecolare del ROKS 50/50: geometrie e frequenze

La prova mantiene fissi peso 50/50, λ=0,9, scaling dei gap pari a uno, assenza di offset e riferimenti DFT congelati. I due componenti sono ottimizzati separatamente con GMF e seguiti tramite APOC. Non viene adattato alcun parametro alle geometrie o alle frequenze di riferimento.

## Risultato del percorso principale

| Molecola | Esito geometrico | Controllo dell'Hessiana nucleare analitica |
|---|---|---|
| HNO | arresto prima della convergenza, gradiente massimo 1,69×10⁻² Ha/bohr | stato vicino a una variazione dell'indice orbitale; nessun minimo certificato |
| Acetaldeide | punto stazionario, gradiente massimo 1,43×10⁻⁶ Ha/bohr | due frequenze immaginarie: −975,87 e −598,50 cm⁻¹ |
| Maleimmide | punto quasi stazionario, gradiente massimo 1,14×10⁻⁵ Ha/bohr | tre frequenze immaginarie: −817,91, −335,49 e −303,70 cm⁻¹ |

Nessuno dei tre percorsi principali supera il gate di minimo. Le frequenze negative sono troppo grandi per essere attribuite a rumore numerico. Per acetaldeide e maleimmide le Hessiane analitiche concordano con differenze indipendenti dei gradienti lungo una direzione nucleare casuale: errori massimi al passo 10⁻⁴ bohr rispettivamente **4,81×10⁻¹⁰** e **2,33×10⁻⁹ Ha/bohr²**, con convergenza quadratica al dimezzamento del passo.

Questi controlli verificano le derivate della superficie scelta. Non dimostrano che quella superficie riproduca correttamente la fisica del minimo eccitato.

## Continuità elettronica e continuazioni

Nel punto terminale accettato di HNO, gli indici orbitali conservati sono ancora (0,0), ma la minima curvatura assoluta del componente completo è circa 2,38×10⁻⁶ Ha. I tentativi successivi non soddisfano più contemporaneamente stazionarietà e indice richiesto. Il gradiente nucleare resta grande: le frequenze calcolabili nell'ultimo punto accettato non sono uno spettro di equilibrio.

L'acetaldeide raggiunge inizialmente un punto planare. Le due continuazioni previste, con perturbazioni ±0,02 Å RMS lungo il modo più instabile, si arrestano entrambe al limite del controllo CSF rispetto al riferimento iniziale. Il gradiente massimo resta circa 1,63×10⁻² Ha/bohr. Overlap appena inferiori a 0,7 vengono rifiutati; la soglia non viene abbassata. Si tratta di un limite di questo protocollo di continuità: da solo non dimostra che un minimo della superficie non esista.

Le due continuazioni della maleimmide si arrestano entrambe dopo 24 iterazioni, con gradiente massimo circa 1,12×10⁻² Ha/bohr, e non producono un minimo qualificato. I tempi delle due riprese sono circa 279 e 270 s. Nessuna frequenza di equilibrio viene assegnata a questi punti non stazionari.

L’audit finale dei checkpoint trova overlap minimi rispetto al riferimento iniziale di circa **0,70004** per le continuazioni dell’acetaldeide e **0,70020** per quelle della maleimmide. Entrambe si avvicinano quindi al confine imposto dal protocollo, con gradienti ancora non piccoli. Questo identifica il controllo globale sull’ancora come primo limite da riesaminare con un nuovo protocollo di tracking; non prova l’assenza matematica di minimi sulla superficie.

Le perturbazioni riutilizzano l'Hessiana del punto stazionario come seme del modello geometrico. Non ripetono le calibrazioni DFT né i lavori elettronici già completati. I riferimenti iniziali e accettati dei due Hamiltoniani restano distinti.

## Accuratezza strutturale: confronto diagnostico

Il riferimento scelto è **CCSDR(3)/def2-TZVPP**, per i medesimi stati singoletti n→π*, dalle Tabelle S19 e S27 di [Budzák, Scalmani e Jacquemin, JCTC 2017](https://doi.org/10.1021/acs.jctc.7b00921). I valori e le coordinate del supplemento sono stati recuperati dal [deposito ACS Figshare](https://api.figshare.com/v2/articles/5659315) e verificati anche visivamente.

| Parametro | Riferimento | Punto principale 50/50 |
|---|---:|---:|
| Acetaldeide C=O / Å | 1,331 | 1,3184 |
| Acetaldeide C–C / Å | 1,508 | 1,4884 |
| Acetaldeide piramidalizzazione η / gradi | 38,7 | circa 0 |
| Maleimmide C=O / Å | 1,248 | 1,2494 |
| Maleimmide N–C / Å | 1,404 | 1,3775 |
| Maleimmide C–C singolo / Å | 1,432 | 1,4189 |
| Maleimmide C=C / Å | 1,386 | 1,3693 |

L'accordo della sola distanza C=O della maleimmide non compensa le instabilità vibrazionali e gli altri errori strutturali. Le continuazioni dell'acetaldeide arrivano a C=O circa 1,3356 Å e η circa 40,12°, ma rimangono non stazionarie: non vengono conteggiate come geometrie di equilibrio accurate.

Per HNO sono stati raccolti i riferimenti armonici e anarmonici di [Davis e Fortenberry, JCTC 2021](https://doi.org/10.1021/acs.jctc.1c00307), Tabelle 7 e 8. Le frequenze armoniche del riferimento (T)+EOM/aug-cc-pVTZ sono 990,2, 1458,8 e 3140,5 cm⁻¹; i parametri geometrici pubblicati sono mediati vibrazionalmente, R₀, e non vengono confusi con valori di equilibrio rₑ. In assenza di un minimo HNO qualificato, non si fornisce una statistica di accuratezza delle frequenze del candidato.

## Confronto verticale con TDA su orbitali GFN2

Sono riutilizzati i controlli TDA completati sui **propri orbitali GFN2 corretti**, con il rispettivo kernel. Non sono TDA su orbitali DFT. La colonna TDA media è il controllo aritmetico dei due risultati, non la risposta di un unico funzionale medio.

| Molecola | TBE verticale / eV | ROKS 50/50 / eV | TDA completo / eV | TDA spettrale / eV | Media dei controlli TDA / eV |
|---|---:|---:|---:|---:|---:|
| HNO | 1,743 | 1,4848 | 3,4498 | 3,4171 | 3,4334 |
| Acetaldeide | 4,304 | 4,2772 | 6,2172 | 5,6910 | 5,9541 |
| Maleimmide | 3,810 | 3,6432 | 4,6295 | 4,5375 | 4,5835 |

Sono controlli verticali già disponibili; non costituiscono una nuova validazione cieca. Il vantaggio verticale del 50/50 rimane, ma non stabilisce l'accuratezza della sua superficie nucleare.

## Costi e provenienza dei tempi

I dati di tempo restano separati per fase e per lavoro. Per acetaldeide/maleimmide, gli SCF DFT dei riferimenti già completati richiesero rispettivamente circa 1,69/20,75 s; la costruzione della risposta completa circa 1,69/22,29 s. Questi costi non vengono conteggiati come zero solo perché in questa prova sono stati riutilizzati.

Il percorso principale della maleimmide, compresa la verifica finale analitica e le differenze direzionali, ha richiesto circa 328 s nella ripresa registrata, oltre alle fasi preparatorie precedenti. Il dato è riferito a un percorso che fallisce il gate di minimo. Non viene confrontato mediante un rapporto di accelerazione con i soli solver TDA verticali: questi ultimi richiedevano circa 0,52 s complessivi per i due componenti della maleimmide, escludendo la preparazione e qualsiasi ottimizzazione geometrica.

Non è dimostrato un vantaggio di tempo del prototipo 50/50. Un confronto fra ottimizzazioni TDA e ROKS richiederebbe prima superfici entrambe qualificate e un benchmark omogeneo.

## Correzioni emerse dalla prova completa

La prova ha individuato e corretto un errore nel dispatch della Hessiana finale di LINK: il percorso nativo eccitato era classificato come disponibile solo con gradienti, e il suo fallback richiamava il programma fondamentale. Le frequenze finali positive originariamente scritte per l'acetaldeide da quel percorso sono **invalidate**. I valori fisici riportati qui provengono dall'Hessiana analitica del 50/50 e dalla sua verifica indipendente.

Il dispatch finale ora richiede direttamente la Hessiana analitica composita. Inoltre, il fallback numerico della TDA conserva l'Hamiltoniano eccitato e non può sostituirlo con quello fondamentale. È stata corretta anche la ripresa di un'ancora iniziale CSF che contiene solo energia: l'ancora può essere ripristinata senza gradiente, mentre il checkpoint del punto accettato continua a richiederlo.

**47 test superati su Oracle**, compresi i tre casi di regressione per queste correzioni. Gli errori iniziali di configurazione del servizio cartesiano e i contratti preparatori superati sono conservati. I punti elettronici e le Hessiane salvate sono stati recuperati; le calibrazioni DFT non sono state rilanciate.

## Decisione

Il 50/50 resta un candidato **verticale**. Questa prova non autorizza a promuovere l'attuale estensione a riferimento congelato come metodo generale per geometrie e frequenze eccitate. Occorre distinguere il tracciamento della stessa natura elettronica durante grandi rilassamenti dal vincolo di overlap globale con la geometria iniziale, e verificare la dipendenza geometrica della correzione non locale. Questi sarebbero nuovi test separati: non vengono usati per modificare retroattivamente i gate della prova presente.

Tutti i lavori elettronici, le ottimizzazioni, le frequenze e le diagnostiche numeriche molecolari sono stati eseguiti su **oracle.sns.it** verificato. Sul Mac sono stati svolti solo modifiche, lettura delle fonti, controlli sintattici, trasferimenti e checksum. Il protocollo è in `protocol.json`; risultati e checkpoint in `results/summary.json` e nelle sottodirectory molecolari.
