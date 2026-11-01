"""Prepare the frozen candidate on the prospective molecules."""
from pathlib import Path
import sys
ROOT=Path('/home/enzo/MOUSE-frontier-cisd-20261101');CAM=Path('/home/enzo/MOUSE-cam-response-20261101');NATIVE=Path('/home/enzo/MOUSE-native-reference-20261101')
sys.path.insert(0,str(CAM))
from anchor import *
from anchor import anchor
from calibrate import calibrate
from full_reference import full_reference
from native_batch import install_native_batch
from heldout_predict import predict_molecule
import heldout_predict as hp,nosi_coherent as nc,nosi_projected as npj
import importlib.util
spec=importlib.util.spec_from_file_location('native_engine',NATIVE/'run.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
if __name__=='__main__':
    target=CAM/'data/prospective_cisd'
    if not target.exists():target.symlink_to(ROOT/'data/prospective_cisd')
    with threadpool_limits():
        for row in json.loads((ROOT/'protocol.json').read_text())['prospective_cases']:
            name=row['molecule'];cohort='prospective_cisd'
            anchor(cohort,name);full_reference(cohort,name,prepare_only=True);calibrate(cohort,name);full_reference(cohort,name)
            install_native_batch(prepare(cohort,name),CAM/cohort/name)
            engine.prepare_native(cohort,name);hp.DEST=NATIVE;nc.DEST=NATIVE;nc.HERE=NATIVE/'nosi_coherent';npj.HERE=nc.HERE
            predict_molecule(cohort,name);nc.run(cohort,name);npj.projected(cohort,name)
