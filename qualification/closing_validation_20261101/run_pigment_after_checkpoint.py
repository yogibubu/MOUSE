"""One-shot remote continuation; wait for a saved result before taking its slot."""
from pathlib import Path
import socket,argparse,time,subprocess,sys
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--wait-result',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
while not (HERE/a.wait_result).exists():time.sleep(10)
subprocess.run([sys.executable,'-u',*a.command],cwd=HERE,check=True)
