"""Post-validation from retained densities; never reruns a completed SCF."""
from anchor import *

def run():
    for cohort,name in [('independent','Criegee_Intermediate'),('independent_challenge','Furofuran')]:
        out=DEST/cohort/name;src=OUT/cohort/'predictions'/name;s=loadpoint(src/'native_ground.npz').reference_state
        with np.load(out/'embedded_ground.npz') as a:p=a['density'].copy();f=a['fock'].copy();c=a['coefficients'].copy()
        commutator=float(np.max(abs(s.overlap@p@f-f@p@s.overlap)))
        charge_error=abs(float(np.trace(s.overlap@p))-s.electron_count)
        idem=float(np.max(abs(p@s.overlap@p-2*p)))
        assert commutator<1e-7 and charge_error<1e-8 and idem<1e-7
        write(out/'density_stationarity_certificate.json',dict(host=HOST,metric_commutator_max=commutator,
            electron_count_error=charge_error,restricted_density_idempotency_error=idem,
            checkpoint_sha256=sha(out/'embedded_ground.npz'),source_sha256=sha(__file__),new_scf_performed=False))

if __name__=='__main__':run()
