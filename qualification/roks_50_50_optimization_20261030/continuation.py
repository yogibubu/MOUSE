"""Two predeclared symmetry-breaking branches, same frozen PES and CSF guards."""
import sys,importlib.util,traceback,json,shutil
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030')
spec=importlib.util.spec_from_file_location('campaign',BASE/'source/qualification/run_campaign_recovered.py');campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)
np=campaign.np
with campaign.threadpool_limits(limits=2):
    for name in sys.argv[1:]:
        root=BASE/'results'/name
        with np.load(root/'main/final.npz') as d:xyz=d['coordinates_angstrom'];h=d['hessian']
        atoms=campaign.prepare(name)['atoms'];modes=campaign._exact_cartesian_normal_modes(atoms,xyz,h)
        k=int(np.argmin(modes.frequencies_cm));assert modes.frequencies_cm[k]<-20.
        direction=modes.cartesian_directions[k].copy();direction/=np.sqrt(np.mean(np.sum(direction*direction,axis=1)))
        for sign,label in [(-1,'negative_mode_minus'),(1,'negative_mode_plus')]:
            try:
                dest=root/label/'roks_50_50'
                if not dest.exists():shutil.copytree(root/'main/roks_50_50',dest)
                geometry=xyz+sign*.02*direction
                campaign.write(root/label/'branch_protocol.json',dict(parent='main',mode_index=k,frequency_cm=float(modes.frequencies_cm[k]),rms_displacement_angstrom=.02,sign=sign,parameters_unchanged=True))
                campaign.optimize(name,label,geometry)
            except Exception as exc:
                campaign.write(root/label/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
                print('BRANCH FAILURE',name,label,traceback.format_exc(),flush=True)
