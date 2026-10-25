#!/usr/bin/env python3
"""Build all internal wheels locally, without GitHub Actions."""
from pathlib import Path
import argparse,subprocess,sys,shutil
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--download-dependencies',action='store_true',help='Also collect external Python dependency wheels for this Python/platform')
a=p.parse_args();root=Path(__file__).resolve().parents[1];out=root/'dist'/'wheels';out.mkdir(parents=True,exist_ok=True)
projects=sorted(c.parent for c in (root/'packages').glob('*/pyproject.toml'))+[root]
for project in projects:
 subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(out),str(project)],check=True)
if a.download_dependencies:
 subprocess.run([sys.executable,'-m','pip','download','--only-binary=:all:','--find-links',str(out),'--dest',str(out),'mouse-platform==0.1.0'],check=True)
print(out)
