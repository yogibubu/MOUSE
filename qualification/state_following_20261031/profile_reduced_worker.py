"""Bounded Oracle-only profile, preserving completed continuation checkpoints."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import cProfile, fcntl, json, pathlib, pstats, signal, socket, sys, time
assert socket.getfqdn() == 'oracle.sns.it'
R = pathlib.Path('/home/enzo/MOUSE-state-following-20261031')
P = pathlib.Path('/home/enzo/MOUSE-publication-validation-20261030')
sys.path.insert(0, str(R))
import reduced_worker as worker
from threadpoolctl import threadpool_limits

if __name__ == '__main__':
    name, sid = sys.argv[1:3]
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    lock = (P/'blind/references/compute.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = R/'performance'/name/sid
    out.mkdir(parents=True, exist_ok=True)
    def stop(*_):
        raise TimeoutError('bounded profiling interval completed')
    signal.signal(signal.SIGALRM, stop)
    profiler = cProfile.Profile()
    start = time.monotonic()
    status = 'completed'
    try:
        signal.alarm(duration)
        with threadpool_limits(limits=2):
            profiler.runcall(worker.run, name, sid)
    except Exception as exc:
        status = type(exc).__name__ + ': ' + str(exc)
    finally:
        signal.alarm(0)
        profiler.dump_stats(str(out/'profile.pstats'))
        with (out/'profile.txt').open('w') as f:
            pstats.Stats(profiler, stream=f).strip_dirs().sort_stats('cumulative').print_stats(45)
        report = dict(host=socket.getfqdn(), molecule=name, state=sid,
                      elapsed_seconds=time.monotonic()-start, status=status,
                      model_unchanged=True, checkpoints_preserved=True,
                      reference_excitation_labels_read=False)
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
