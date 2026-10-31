"""Oracle-only frozen-orbital development diagnostic; no target labels loaded."""
import sys
from pathlib import Path
R=Path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration/final_validation')
sys.path.insert(0,str(R))
from common import *
from matrix_gfn2.relative_pair_roks import load_relative_pair_roks
from pair_cas import evaluate_pair_cas
import fcntl,traceback
DEST=Path('/home/enzo/MOUSE-transferability-20261101')

def molecular(cohort,name):
    src=R/cohort/'predictions'/name
    rows=[r for r in json.loads((R/cohort/'manifest.json').read_text())['states'] if r['molecule']==name]
    raw=loadpoint(src/'native_ground.npz');s=canonical_state(raw)
    calc=calculator(s);xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR
    ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
    responses=[];grounds=[]
    for i,profile in enumerate(['full_response','qualified_spectral_response']):
        p=src/profile/'anchor/model.npz'
        with np.load(p) as a:d={k:a[k].copy() for k in ['operators','charge','spin','linear','q0']}
        response=FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,d,sha(p))
        assert (src/f'grounds/{i}/certificate.json').exists(),'Reuse cached grounds only'
        responses.append(response);grounds.append(ground(src/f'grounds/{i}',calc,xyz,s,response))
    calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC'))
    contexts=[prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=r) for r in responses]
    no=s.electron_count//2;gc=[]
    for context,g in zip(contexts,grounds):
        _,c=eigh(g.fock[0],s.overlap)
        gc.append(evaluate_pair_cas(context,c,no))
    for row in rows:
        dest=DEST/cohort/name/row['id']/'result.json'
        if dest.exists():continue
        old=json.loads((src/row['id']/'result.json').read_text())
        if old['status']=='not_representable':
            write(dest,dict(id=row['id'],molecule=name,status='not_representable',basis_unchanged=True));continue
        projection=json.loads((src/row['id']/'projection.json').read_text())
        if projection['rank']!=1:
            write(dest,dict(id=row['id'],molecule=name,status='outside_pair_diagnostic',rank=projection['rank']));continue
        joint=load_relative_pair_roks(src/row['id']/'stationary.npz',request_hash='independent:relative',require_gradient=False)
        parents=[]
        for context,g,gs,component in zip(contexts,grounds,gc,joint.components):
            opened=np.flatnonzero(component.occupations[0]!=component.occupations[1]);assert len(opened)==2
            hole=int(opened[opened<no][0]);particle=int(opened[opened>=no][0])
            ex=evaluate_pair_cas(context,component.common_coefficients,no,hole=hole,particle=particle)
            shift_exc=ex.energies[ex.target_root]-ex.singlet_diagonal
            shift_ground=gs.energies[0]-g.energy_hartree
            parents.append(dict(gap_ev=(ex.energies[ex.target_root]-gs.energies[0])*EV,
                baseline_gap_ev=(ex.singlet_diagonal-g.energy_hartree)*EV,
                excited_shift_ev=shift_exc*EV,ground_shift_ev=shift_ground*EV,
                ground_pair=[no-1,no],excited_pair=[hole,particle],
                target_root=ex.target_root,target_weight=ex.target_weight,target_margin=ex.target_margin,
                ground_eigenvalues=gs.energies,excited_eigenvalues=ex.energies,
                excited_matrix=ex.matrix,ground_matrix=gs.matrix))
        qualified=all(p['target_weight']>.5 and p['target_margin']>.05 for p in parents)
        write(dest,dict(id=row['id'],molecule=name,status='diagnostic_complete',assignment_qualified=qualified,
            excitation_ev=.5*sum(p['gap_ev'] for p in parents),baseline_excitation_ev=.5*sum(p['baseline_gap_ev'] for p in parents),
            parents=parents,energy_labels_used=False,protocol_sha256=sha(DEST/'protocol.json'),source_sha256=sha(__file__)))
        print(name,row['id'],'complete',flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        for cohort,name in [('independent','Criegee_Intermediate'),('independent','Benzoxadiazole'),('independent','Hexatriene'),('independent_challenge','Furofuran'),('independent_state_challenge','Benzoxadiazole')]:
            try:molecular(cohort,name)
            except Exception:write(DEST/cohort/name/'diagnostic_error.json',dict(traceback=traceback.format_exc()))
