"""Aggregate only paired qualified states; preserve negative validation results."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import json,sys,platform,hashlib
from spin_kernel_probe import OLD,write,digest
R=BASE/'results'

def records(path):return json.loads(path.read_text())['records']
def metrics(rows,field):
 e=[r[field]-r['tbe_ev'] for r in rows if r.get(field) is not None]
 return dict(count=len(e),mae_ev=sum(map(abs,e))/len(e),maxabs_ev=max(map(abs,e))) if e else None
rows=[]
for c in json.loads((R/'full_reference_inputs/frozen_relaxation/plan.json').read_text())['cases']:
 m=c['name'];hold=(BASE/'holdout_data'/m/'case.json').exists();prefix=R/'holdout' if hold else R
 baseline=BASE/'holdout_data/frozen_relaxation'/m/'result.json' if hold else OLD/'frozen_relaxation'/m/'result.json'
 for t in c['targets']:
  sid=t['id'];old=next(x for x in records(baseline) if x['id']==sid)
  response=next(x for x in records(prefix/'response_kernel'/m/'result.json') if x['id']==sid)
  bh=next(x for x in records(prefix/'bhandhlyp'/m/'result.json') if x['id']==sid)
  if not all(x['qualified'] for x in [old,response,bh]):continue
  row=dict(molecule=m,state=sid,cohort='holdout' if hold else 'development',tbe_ev=t['tbe_ev'],
   native_roks_ev=old.get('se_relaxed_ev',old.get('se_ev')),old_hybrid_ev=old['hybrid_ev'],
   frozen_candidate_ev=bh['frozen_ev']+response['relaxation_ev'],matched_roks_ev=response['raw_roks_ev'],
   roks_seconds=response.get('optimization_seconds'),roks_index=response['index'])
  # Original frozen-relaxation records store the original SE excitation as roks_ev.
  for k in ['se_excitation_ev','raw_roks_ev','roks_ev']:
   if row['native_roks_ev'] is None and k in old:row['native_roks_ev']=old[k]
  full=R/'full_reference'/m/sid/'result.json'
  row['full_reference_status']='not qualified'
  if full.exists():
   f=json.loads(full.read_text())
   if f['qualified']:row.update(full_reference_status='qualified',relaxed_dft_ev=f['excitation_ev'],dft_residual=f['full_orbital_residual'],dft_csf_overlap=f['seed_csf_overlap'])
  tdap=R/'tda'/row['cohort']/m/'result.json'
  if tdap.exists():
   td=json.loads(tdap.read_text());tr=next(x for x in td['rows'] if x['state']==sid)
   for label in ['native','matched']:
    if tr[label]['qualified']:row[label+'_tda_ev']=tr[label]['energy_ev']
   row['tda_seconds']=td['matched_tda_solver_seconds']
  ap=R/'relaxed_anchors'/m/sid/'validation.json'
  if ap.exists():row['local_anchor']=json.loads(ap.read_text())
  ohno=R/'ohno_tda_control'/m/'result.json'
  if ohno.exists():
   o=next(x for x in json.loads(ohno.read_text())['rows'] if x['state']==sid)
   if o['qualified']:row['ohno_tda_ev']=o['energy_ev']
  rows.append(row)
# Recover original SE values from original source files where required.
for row in rows:
 if row['native_roks_ev'] is None:
  prefix=BASE/'holdout_data' if row['cohort']=='holdout' else OLD
  old=next(x for x in records(prefix/'frozen_relaxation'/row['molecule']/'result.json') if x['id']==row['state'])
  row['native_roks_ev']=old.get('se_relaxed_ev',old.get('se_roks_ev'))
cohorts={}
for cohort in ['development','holdout']:
 rr=[r for r in rows if r['cohort']==cohort]
 cohorts[cohort]={f:metrics(rr,f) for f in ['native_roks_ev','old_hybrid_ev','frozen_candidate_ev','matched_roks_ev','native_tda_ev','matched_tda_ev','ohno_tda_ev','relaxed_dft_ev']}
full_paired={}
for co in cohorts:
 rr=[r for r in rows if r['cohort']==co and 'relaxed_dft_ev' in r]
 full_paired[co]={f:metrics(rr,f) for f in ['native_roks_ev','old_hybrid_ev','frozen_candidate_ev','matched_roks_ev','relaxed_dft_ev']}
paired={}
for co in cohorts:
 rr=[r for r in rows if r['cohort']==co and 'native_tda_ev' in r and 'matched_tda_ev' in r]
 paired[co]={f:metrics(rr,f) for f in ['native_roks_ev','matched_roks_ev','native_tda_ev','matched_tda_ev']}
hold=[r for r in rows if r['cohort']=='holdout'];mae0=metrics(hold,'old_hybrid_ev');mae1=metrics(hold,'frozen_candidate_ev')
regression=max(abs(r['frozen_candidate_ev']-r['tbe_ev'])-abs(r['old_hybrid_ev']-r['tbe_ev']) for r in hold)
gate=bool(len(hold)>=3 and {r['state'] for r in hold}>={'n_pi','pi_pi'} and mae1['mae_ev']<mae0['mae_ev'] and mae1['maxabs_ev']<=mae0['maxabs_ev'] and regression<=.25)
summary=dict(host=HOST,rows=rows,cohorts=cohorts,tda_roks_paired=paired,full_reference_paired=full_paired,holdout_gate_passed=gate,maximum_holdout_regression_ev=regression,
 holdout_protocol_sha256=digest(R/'holdout_protocol.json'),new_default=False,
 relaxed_anchor_scope='local model after failed holdout; not an independent accuracy validation',
 full_dft_hessian_index_certified=False,full_dft_nuclear_derivatives_available=False,
 unqualified_full_references=[dict(molecule=r['molecule'],state=r['state']) for r in rows if 'relaxed_dft_ev' not in r])
write(R/'summary.json',summary)
lines=['# ROKS, risposta DFT e validazione molecolare — 30 ottobre 2026','',
'Il codice e le derivate sono verificati. La correzione economica selezionata sui dati di sviluppo **fallisce** sulle nuove molecole e non diventa il metodo predefinito. L’ancoraggio a energie DFT completamente rilassate è disponibile come modello locale esplicito; non costituisce una nuova validazione cieca.',
'', '## Accuratezza e generalizzazione','',
'La scelta BH&HLYP (energia congelata) + risposta PBE0 di carica/spin proiettata su GFN2 è stata fissata prima dei nuovi calcoli. λ=0,9 moltiplica una sola volta lo scambio HF nella risposta di riferimento; non moltiplica gap o energie ROKS. Nessun λ dipendente dall’estensione è stato promosso.',
'', '| Insieme | Modello | N | MAE (eV) | Errore massimo (eV) |','|---|---|---:|---:|---:|']
for co in cohorts:
 for f,label in [('native_roks_ev','ROKS SE originale'),('old_hybrid_ev','PBE0 congelato + rilassamento SE'),('frozen_candidate_ev','Candidato BH&HLYP congelato + risposta corretta'),('matched_roks_ev','ROKS SE con risposta corretta, senza ancoraggio'),('native_tda_ev','TDA nativa su orbitali GFN2'),('matched_tda_ev','TDA con risposta corretta, su orbitali GFN2'),('ohno_tda_ev','Ohno J/K, profilo PBE0 λ=0,9 su orbitali GFN2'),('relaxed_dft_ev','ROKS BH&HLYP completamente rilassato (controllo successivo)')]:
  a=cohorts[co][f]
  if a:lines.append(f"| {co} | {label} | {a['count']} | {a['mae_ev']:.4f} | {a['maxabs_ev']:.4f} |")
lines+=['', '| Insieme appaiato | Modello | N comune | MAE (eV) |', '|---|---|---:|---:|']
for co,values in paired.items():
 for f,a in values.items():
  if a:lines.append(f"| {co} | {f} | {a['count']} | {a['mae_ev']:.4f} |")
lines+=['', 'Le statistiche TDA/ROKS su identiche righe qualificate sono conservate anche in summary.json (tda_roks_paired). Le medie con numerosità diversa non sono confronti appaiati.', '',f'Esito del gate congelato: **{gate}**. Peggioramento individuale massimo: {regression:.4f} eV (limite 0,25 eV). Le medie DFT rilassate possono includere meno stati: confrontare sempre le righe comuni della tabella seguente.',
'', 'Il risultato senza offset energetico è un’analisi successiva del modello, non il superamento del gate del candidato congelato: richiede un’ulteriore prova indipendente prima di promuoverlo. Il pirrolo resta escluso: il riferimento B2 ha proiezione di transizione di circa 0,48 nella base compatta GFN2, sotto la soglia fissata 0,7. Non sono state cambiate simmetrie o soglie. La citosina n→π* resta esclusa per assegnazione ambigua nel precedente protocollo. L’uracile π→π* non supera il gate di identità TDA e resta fuori dai confronti TDA/ROKS appaiati.',
'', '| Molecola | Stato | QUEST (eV) | ROKS SE | TDA GFN2 | Candidato congelato | DFT rilassato |', '|---|---|---:|---:|---:|---:|---:|']
def fmt(x):return '—' if x is None else f'{x:.4f}'
for r in rows:lines.append(f"| {r['molecule']} | {r['state']} | {r['tbe_ev']:.3f} | {fmt(r['native_roks_ev'])} | {fmt(r.get('native_tda_ev'))} | {r['frozen_candidate_ev']:.4f} | {fmt(r.get('relaxed_dft_ev'))} |")
lines+=['','## Costo: confronto coerente','',
'La TDA denominata nativa nella tabella usa il kernel di risposta SCC GFN2. La TDA corretta usa U ottenuto dalla risposta DFT. Il controllo separato Ohno J/K usa il profilo PBE0 con coefficiente finale di scambio 0,25×0,9=0,225 e filtro spaziale 0,5, applicato agli orbitali/gap GFN2: questi ultimi non contengono una frazione HF 0,25; è una convenzione del kernel di riferimento, non una proprietà GFN2. I risultati mostrano che trasferire questo kernel senza un bilanciamento coerente dei gap produce errori grandi; non si confonde tale controllo con una TDA dello stesso funzionale SE. Le matrici TDA usano davvero gli orbitali canonici GFN2 dello stato fondamentale. I gap DFT mappati sono un esperimento distinto, salvato nei risultati grezzi, che produce errori grandi e non viene selezionato. Nessun confronto con TDA su orbitali DFT viene presentato come equivalente.',
'', '| Nuova molecola | TDA, costruzione condivisa + diagonalizzazione (s) | ROKS risposta corretta da orbitali canonici (s) |', '|---|---:|---:|']
for r in hold:lines.append(f"| {r['molecule']} | {r['tda_seconds']:.5f} | {r['roks_seconds']:.5f} |")
lines+=['',
'Questi tempi escludono la preparazione DFT, che è un costo aggiuntivo del modello corretto. La TDA produce più radici per matrice; ROKS ottimizza uno stato. I tempi di sviluppo ROKS partono da checkpoint rilassati e non sono confronti di avvio a freddo. Non si osserva un vantaggio generale di velocità ΔSCF.',
'', '## Modello variazionale e derivate','',
'Per ciascuno stato qualificato si salva un modello locale: E(R)=E_ROKS,GFN2+risposta_congelata(R)+C. W di spin, U di carica, popolazioni di riferimento e C restano fissi. La metrica AO e la densità dipendono dalla geometria. Per kind=relaxed_vertical, C riproduce a R0 l’energia di eccitazione ROKS DFT completamente rilassata. Una costante di stato non migliora le forze; queste restano quelle del modello SE corretto.',
'',
'Il singoletto è una CSF S=0 a due determinanti con orbitali comuni; si ottimizza E_S=2E_misto−E_tripletto. Non si rivendica la proiezione esatta dell’energia di un determinante UKS rotto. La ricerca GMF ammette curvature elettroniche negative; l’indice è verificato contro gli autovalori dell’Hessiana orbitale SE. Per i riferimenti DFT si verificano residuo orbitale ≤10⁻⁷ e overlap CSF ≥0,7, senza certificare l’indice completo.',
'',
'L’Hessiana nucleare è analitica e include il complemento di Schur della risposta orbitale, i contributi nativi GFN2, spin, carica e metrica. Non usa geometrie spostate. Le differenze finite sono controlli indipendenti: HNO, tutte le 9 colonne, errore massimo 1,55×10⁻⁶ Ha/bohr²; uracile, Hessiane complete 36×36, controlli direzionali casuali e stretching C=O, errori massimi 4,55×10⁻⁶ (nπ*) e 2,65×10⁻⁶ (ππ*) Ha/bohr². Queste sono derivate esatte del modello congelato, non Hessiane DFT e non derivate di parametri ricalibrati a ogni geometria.',
'',
'APOC conserva gli overlap coerenti delle CSF; LINK usa lo stesso modello per energie, gradienti e Hessiane, con fingerprint del funzionale e checksum del file nel restart. I punti accettati aggiornano il riferimento. Test: 43 passati su Oracle; controllo molecolare reale LINK/APOC in results/link_molecular_validation.json.',
'', '## Uso e limiti','',
'I file results/relaxed_anchors/MOLECOLA/STATO/model.npz contengono il modello completo senza pickle; central.npz il checkpoint, analytic.npz gradiente/Hessiana/curvature, backend.json la configurazione LINK. I percorsi nel backend sono quelli Oracle e vanno aggiornati insieme al file modello quando si trasferisce il lavoro. Il checksum del contenuto resta valido. Ogni validation.json dichiara lo scopo e verifica energia, risposta orbitale, traslazioni e indice.',
'',
'I riferimenti DFT usano BH&HLYP/aug-cc-pVDZ cartesiana, griglia 3 e ottimizzazione diretta degli orbitali. Il congelamento iniziale degli orbitali aperti è seguito dal rilascio di tutte le rotazioni attive. Checkpoint già completati sono riutilizzati. Un riferimento che perde identità non viene sostituito da uno stato più basso. Le eventuali failure.json conservano i tentativi, anche quando una successiva ripresa riesce: result.json qualificato è l’esito finale.',
'',
'Conclusione scientifica: riuso del programma di stato fondamentale e derivate analitiche sono dimostrati; una correzione semiempirica universale dell’errore n→π* non è dimostrata. Gli ancoraggi rilassati pagano un calcolo DFT per stato e non possono essere usati come prova che la sola approssimazione SE abbia raggiunto quell’accuratezza. Il metodo predefinito resta invariato.',
'', '## Lavori e idee di Levi','',
'[Levi, Ivanov e Jónsson, Faraday Discussions 2020](https://doi.org/10.1039/D0FD00064G), [Schmerwitz, Levi e Jónsson, JCTC 2023, Generalized Mode Following](https://doi.org/10.1021/acs.jctc.3c00178) e [Freeze-and-Release Direct Optimization, JCTC 2026](https://doi.org/10.1021/acs.jctc.5c01974) completano i riferimenti metodologici. [Levi, Ivanov e Jónsson, JCTC 2020, Variational Density Functional Calculations of Excited States via Direct Optimization](https://doi.org/10.1021/acs.jctc.0c00597): ottimizzare direttamente le rotazioni orbitali verso soluzioni stazionarie eccitate. [Documentazione ufficiale GPAW](https://gpaw.readthedocs.io/documentation/do/do.html): ricerca di saddle e preparazione con orbitale buco/particella congelati prima del rilascio. Queste idee motivano la ricerca e il controllo di identità; non forniscono una parametrizzazione universale GFN2.',
'', '## Tracciabilità','',
'Tutti i lavori elettronici e numerici sono stati eseguiti su oracle.sns.it dopo verifica del nome host. scripts, checkpoint, colonne di risposta, matrici TDA, gradienti/Hessiane e protocolli sono conservati. summary.json mantiene i valori completi; provenance.json registra ambiente e hash SHA256. Il laptop ha svolto solo orchestrazione, modifica del testo/codice e trasferimenti.']
(BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
write(R/'environment.json',dict(host=HOST,python=sys.version,platform=platform.platform()))
print(json.dumps(dict(cohorts=cohorts,gate=gate,full_count=sum('relaxed_dft_ev' in r for r in rows)),indent=2))
