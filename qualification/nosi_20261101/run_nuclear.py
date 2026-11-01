from nuclear_assembly import *
import argparse,resource

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cohort');parser.add_argument('molecule');args=parser.parse_args()
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        start=time.monotonic();model=NuclearCandidate(args.cohort,args.molecule);result=model.hessian()
        write(model.destination/'hessian_certificate.json',dict(host=HOST,method='analytic projected-NOSI including complete mixed orbital KKT and generalized-eigenvector response',
            states={k:dict(energy_hartree=v['energy_hartree'],raw_symmetry_error=v['raw_symmetry_error'],translation_error=v['translation_error']) for k,v in result.items()},
            wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            source_sha256=sha(__file__),implementation_sha256=sha(Path(__file__).parent/'nuclear_assembly.py'),protocol_sha256=sha(Path(__file__).parent/'nuclear_protocol.json'),
            independent_energy_derivative_validation='pending; this certificate records internal consistency only'))
