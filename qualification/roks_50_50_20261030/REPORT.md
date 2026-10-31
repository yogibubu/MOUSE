# ROKS 50/50: superficie locale, gradienti e Hessiana nucleare analitica

Il candidato scelto è la media fissa delle **energie di due soluzioni ROKS ottimizzate separatamente**, risposta completa e profilo spettrale qualificato. Non si ottimizza il funzionale medio. Non si aggiungono offset energetici e non si adattano pesi o parametri alle energie di eccitazione.

Per le superfici totali eccitate:

\[
E_{50/50}(R)=\tfrac12 E_{\mathrm{full}}^{\mathrm{ROKS}}(R)+\tfrac12 E_{\mathrm{spectral}}^{\mathrm{ROKS}}(R),\quad
 g_{50/50}=\tfrac12(g_{\mathrm{full}}+g_{\mathrm{spectral}}),\quad
 H_{50/50}=\tfrac12(H_{\mathrm{full}}+H_{\mathrm{spectral}}).
\]

Ciascuna Hessiana è **nucleare, analitica e rilassata**, ottenuta con il proprio complemento di Schur della risposta orbitale. Le differenze finite sono usate esclusivamente nella validazione. L'energia di eccitazione composita sottrae la media delle energie fondamentali, ottimizzate con i rispettivi modelli. I gradienti e le Hessiane restituiti dall'API sono quelli dell'energia totale eccitata: non sono automaticamente le derivate del gap.

## Estensione coerente a geometrie spostate

I modelli non locali precedenti erano soltanto verticali. La nuova classe `FrozenNonlocalResponseReference` definisce un'estensione locale esplicita, mantenendo congelati coefficienti DFT, matrice di risposta e momenti di riferimento. Gli operatori AO sono trasportati con la rotazione propria che allinea ai minimi quadrati i nuclei etichettati della geometria di riferimento alla geometria corrente, dopo la rimozione della traslazione.

Le derivate prime e seconde della rotazione sono ottenute analiticamente dall'equazione di stazionarietà polare; comprendono il caso planare. Si differenziano sia il metric AO sia gli operatori trasportati. La base supportata è s/p; frame lineari o ambigui vengono rifiutati. Questa è una superficie locale a riferimento congelato, non una ricalibrazione DFT a ogni geometria. Non introduce le derivate di parametri DFT ricalcolati lungo il percorso.

Al riferimento, l'estensione riproduce le energie verticali precedenti entro 3×10⁻¹³ eV. λ=0,9 resta già incorporato una sola volta nello scambio della risposta. La nuova classe non scala nuovamente kernel o gap.

## Validazione molecolare remota

Tutti i lavori elettronici, le differenze finite e i test numerici sono stati eseguiti su **oracle.sns.it**, dopo verifica dell'identità. Sul Mac sono stati eseguiti soltanto modifiche ai file, controlli sintattici, trasferimenti e checksum. I checkpoint verticali completati sono stati riutilizzati; nessuna calibrazione DFT è stata ripetuta.

Sono state confrontate tutte le colonne delle Hessiane con differenze finite dei gradienti analitici, a due passi indipendenti, 2×10⁻⁴ e 1×10⁻⁴ bohr. Gli stati spostati sono seguiti mediante overlap CSF con soglia 0,7 e conservazione dell'indice orbitale del componente.

| Molecola, n→π* | Eccitazione 50/50 / eV | Overlap CSF fra componenti | Errore massimo Hessiana 50/50, passo 10⁻⁴ bohr / Ha bohr⁻² |
|---|---:|---:|---:|
| HNO | 1,484776707 | 0,981383 | 1,282×10⁻⁸ |
| Acetaldeide | 4,277243463 | 0,967393 | 1,648×10⁻⁸ |

Per i singoli componenti, gli errori massimi delle Hessiane sono 1,193×10⁻⁸ e 1,999×10⁻⁸ Ha bohr⁻² per HNO; 1,649×10⁻⁸ e 1,647×10⁻⁸ per acetaldeide. Dimezzando il passo, gli errori diminuiscono di circa quattro volte. Il massimo errore dei gradienti contro differenze delle energie al passo più piccolo è 5,1×10⁻⁹ Ha bohr⁻¹.

Superati anche i controlli indipendenti di covarianza di energia, Fock e gradiente sotto rotazione rigida, la somma delle forze, la coppia totale, le identità traslazionali della Hessiana e l'identità rotazionale corretta alle geometrie non stazionarie (`H·δR = δg`, senza imporre falsi modi rotazionali nulli lontano da un minimo). Errori delle identità delle Hessiane inferiori a 6×10⁻¹⁴ nelle rispettive unità atomiche nei controlli effettuati.

L'API pubblica di valutazione e quella della Hessiana composita riproducono i dati dei componenti salvati con scarti inferiori a 3×10⁻¹⁵ nelle rispettive unità atomiche. **39 test passati**, comprendenti derivate scalari indipendenti, risposta orbitale rilassata, persistenza e checksum, combinazione dei componenti e regressioni dei percorsi ROKS/ΔSCF precedenti.

## API e riutilizzo

`matrix_gfn2` esporta `FrozenNonlocalResponseReference`, la persistenza tramite `save_response_reference` / `load_response_reference`, `CompositeROKSResult`, `combine_roks_results`, `evaluate_composite_roks` e `evaluate_composite_roks_analytic_hessian`.

Esempio di prosecuzione dai due stati già qualificati, ordinati **completo, spettrale**:

```python
from matrix_gfn2 import (
    combine_roks_results, evaluate_composite_roks,
    evaluate_composite_roks_analytic_hessian,
)

retained = combine_roks_results(raw_state, spectral_state)
solver_options = [
    dict(orbital_solver="gmf", saddle_index=r.orbital_saddle_index,
         orbital_tolerance=1e-11, maximum_iterations=150)
    for r in retained.components
]
point = evaluate_composite_roks(
    (raw_calculator, spectral_calculator), coordinates_angstrom,
    *raw_state.occupations, (raw_response, spectral_response),
    reference_result=retained, gradient=True,
    component_options=solver_options,
)
hessian = evaluate_composite_roks_analytic_hessian(
    (raw_calculator, spectral_calculator), coordinates_angstrom,
    (raw_response, spectral_response), reference_result=point,
)
```

Ogni componente mantiene la propria CSF singoletto spin-puro e il proprio certificato di indice orbitale. Il risultato composito conserva entrambi: non viene esportato come se fosse una singola funzione d'onda o un unico saddle di un funzionale medio. La continuità fra geometrie resta verificata separatamente per ciascun Hamiltoniano. Il confronto fra i due componenti è un controllo CSF esplicito sulla stessa base e sugli stessi atomi; non disabilita il controllo APOC sul cambio di Hamiltoniano durante una traiettoria.

## Limiti e stato della scelta

Questo lavoro rende utilizzabili e verificabili le derivate della superficie locale 50/50. Non aggiunge una nuova validazione di accuratezza spettroscopica: HNO e acetaldeide erano già dati esposti. Sul precedente gruppo nuovo F, la media resta a MAE 0,076 eV, contro 0,155 eV della correzione ispirata a GX24, che non viene promossa. La correzione universale n→π* rimane aperta.

La nuova API composita è esplicita e sperimentale. Non è stata attivata automaticamente nel backend LINK delle scansioni; quel percorso continua a gestire singoli componenti ROKS. Non sono state eseguite ottimizzazioni o frequenze molecolari di produzione con il composito, né si attribuisce un vantaggio di tempo sulla TDA a questa implementazione densa.

Risultati e checkpoint sono in `results/`; `molecular_validation.py` riproduce i controlli con ripresa dei punti salvati, `api_and_invariance_check.py` controlla API e identità euclidee. `run_protocol.json` conserva l'identità numerica del primo lancio. Gli aggiustamenti successivi hanno corretto esclusivamente il controllo diagnostico fra Hamiltoniani e l'uso di calcolatori separati nel test di rotazione; Hessiane e punti spostati già completati sono stati ripresi, non ricalcolati. Log iniziali e sorgenti eseguite sono conservati.
