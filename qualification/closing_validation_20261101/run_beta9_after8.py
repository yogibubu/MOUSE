import time,subprocess
from pathlib import Path
while not Path("pigment_cid_expansion/Beta_carotene/screening_10/frontier_8/result.json").exists():time.sleep(20)
subprocess.run(["/Users/vincenzobarone/.venvs/matrix/bin/python","-u","run_pigment_cid.py","Beta_carotene","--frontier","9","--screening-frontier","10"],check=True)
