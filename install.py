#!/usr/bin/env python3
"""Install the complete MOUSE distribution into the current Python environment."""
from pathlib import Path
import argparse,subprocess,sys,tomllib

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--offline',action='store_true',help='Use only the release wheel directory')
 a=p.parse_args();root=Path(__file__).resolve().parent
 components=sorted((root/'packages').glob('*/pyproject.toml'))
 wheels=root/'dist'/'wheels'
 command=[sys.executable,'-m','pip','install']
 if a.offline:
  if not wheels.is_dir():p.error('Offline installation requires the platform-specific wheel release')
  command+=['--no-index','--find-links',str(wheels),'mouse-platform==0.1.0']
 elif (wheels/'mouse_platform-0.1.0-py3-none-any.whl').exists():
  command+=['--find-links',str(wheels),'mouse-platform==0.1.0']
 else:
  command += [str(c.parent) for c in components]+[str(root)]
 subprocess.run(command,check=True)
if __name__=='__main__':main()
