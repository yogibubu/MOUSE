# Integrazione automatica ROKS 50/50 in LINK

Il metodo `native_excited_method="roks_50_50"` è integrato nei componenti QM/QM, nelle scansioni dirette `run_qm_scan_points` e nel servizio residente di ottimizzazione geometrica `GeometryEvaluationService`.

LINK ottimizza separatamente le due CSF ROKS singoletto, con gli indici orbitali richiesti per ciascun componente. Restituisce la media fissa 50/50 delle energie totali eccitate, dei gradienti e delle Hessiane nucleari analitiche già rilassate. Non costruisce un funzionale medio, non aggiunge offset e non cambia i pesi durante la scansione.

## Configurazione

```python
from pathlib import Path
from matrix_link.scan import QMScanBackend, run_qm_scan_points

backend = QMScanBackend(
    name="matrix_gfn2",
    electronic_state=1,
    native_excited_method="roks_50_50",
    deltascf_promotion=(hole, particle),  # indici orbitali da zero
    native_roks_composite_response_paths=(
        Path("raw_response.npz"),
        Path("spectral_response.npz"),
    ),
    native_roks_composite_saddle_indices=(raw_index, spectral_index),
    properties=("energy", "gradient", "hessian"),
    processors=2,
)
results = run_qm_scan_points(
    "molecule.xyz", points, backend, run_dir="scan_50_50",
)
```

I due file sono riferimenti `FrozenNonlocalResponseReference` già calibrati per la molecola, ordinati **completo, spettrale**. La configurazione controlla identità atomica, geometria di riferimento, operatori e kernel di carica/spin condivisi; i termini lineari possono differire. I checksum di entrambi i file entrano nel contratto e vengono ricontrollati durante il lavoro. La media non richiede una nuova calibrazione DFT. La produzione di riferimenti per una molecola nuova rimane un passaggio separato.

Per una scansione diretta deve esistere un unico punto a spostamento zero. LINK lo valuta per primo anche se l'ordine fornito è diverso, lo accetta e segue entrambi i componenti da quel riferimento sui punti spostati. I punti spostati non avanzano la storia accettata. L'ordine dei risultati resta quello degli indici dei punti. La stessa configurazione può essere assegnata al servizio di ottimizzazione o a un componente del backend `capped_subtractive`.

## Continuità e ripresa

Ogni componente conserva la propria CSF e il proprio Hamiltoniano; APOC verifica separatamente gli overlap cross-AO con l'ultimo stato accettato e con il riferimento iniziale. È verificato anche l'overlap fra i due componenti sulla stessa geometria, senza disabilitare il controllo APOC sul cambio di Hamiltoniano.

Nell'ottimizzatore soltanto `accept_electronic_state` avanza il riferimento. I tentativi del passo geometrico possono aggiornare i dati di lavoro del calcolatore residente, ma non le CSF accettate. Il calcolo usa occupazioni intere e temperatura elettronica zero.

La directory `roks_50_50_accepted` contiene generazioni di checkpoint per **entrambi i componenti**, sia per il riferimento accettato sia per quello iniziale. Tutti i file di una generazione sono scritti prima di sostituire atomicamente `accepted.json`. Una nuova istanza del componente o del servizio ripristina questi stati, con verifica di contratto, checksum, modello e indice orbitale. Le generazioni precedenti restano conservate.

I contratti di ripresa includono entrambi i file di risposta e i loro contenuti. Un cache hit dell'ottimizzatore verifica comunque i file di risposta; un record della cache su disco privo di CSF viene rivalutato dai checkpoint accettati prima di poter avanzare la storia. Un checkpoint alterato o appartenente a un altro Hamiltoniano viene rifiutato.

## Validazione su Oracle

Tutte le elaborazioni numeriche sono state eseguite su `oracle.sns.it`, dopo verifica dell'identità. Sono passati **44 test**, comprendenti configurazione, persistenza, accettazione atomica delle due CSF, dispatch dell'ottimizzatore, verifica dei file prima dei cache hit e regressioni ROKS/ΔSCF.

La prova molecolare usa il vero ingresso pubblico delle scansioni, con ordine iniziale dei punti `+q, 0, −q`, e il vero servizio di ottimizzazione geometrica. Le energie, i gradienti e le Hessiane sono confrontati con il precedente calcolo 50/50 indipendente dall'adattatore LINK.

| Controllo | HNO | Acetaldeide |
|---|---:|---:|
| Errore energia scansione / Ha | 8,9×10⁻¹⁶ | 1,8×10⁻¹⁵ |
| Errore massimo gradiente / Ha bohr⁻¹ | 2,8×10⁻¹³ | 6,4×10⁻¹⁶ |
| Errore massimo Hessiana / Ha bohr⁻² | 4,7×10⁻¹³ | 5,0×10⁻¹⁵ |
| Errore H·v contro differenze dei gradienti / Ha bohr⁻² | 9,2×10⁻¹⁰ | 1,1×10⁻⁹ |
| Ripresa delle due CSF identica nei coefficienti | sì | sì |
| Tentativo geometrico lascia invariato il checkpoint accettato | sì | sì |
| Checkpoint alterato rifiutato | sì | sì |
| Stato di un altro Hamiltoniano rifiutato prima dell'accettazione | sì | sì |

Anche il servizio di ottimizzazione riproduce le Hessiane precedenti, con scarti inferiori a 5×10⁻¹³ Ha bohr⁻², e riprende correttamente dopo la ricreazione del servizio e la lettura della cache su disco. Sono state validate le valutazioni del servizio, l'accettazione e la ripresa; non viene presentata questa prova come un'ottimizzazione molecolare fino alla convergenza.

## Portata scientifica

L'integrazione conserva la superficie locale con parametri DFT congelati e trasporto co-rotante già validata nel lavoro precedente. Restano il supporto s/p e il rifiuto dei frame lineari o ambigui. I gradienti e le Hessiane sono dell'energia totale eccitata; non sono automaticamente le derivate del gap di eccitazione. L'accuratezza universale n→π* rimane aperta e non viene attribuito un vantaggio di tempo sulla TDA.

`molecular_integration.py` riproduce le prove molecolari; `final_restart_check.py` verifica i controlli finali sui checkpoint completati senza rilanciare lavori elettronici. Risultati, manifesti accettati e log sono in `results/`.
