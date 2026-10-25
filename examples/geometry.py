"""ORACLE perception, SMITH Cartesian coordinates and resident LINK water optimization."""
from pathlib import Path
import json
from matrix_oracle.api import analyze_structure
from matrix_link.native_sycart import optimize_native_sycart_matrix_gfn2
from matrix_smith.definition import write_sonic_build_sections_from_cartesian
root=Path('example-output');root.mkdir(exist_ok=True)
xyz=root/'water.xyz';xyzin=root/'water.xyzin'
xyz.write_text('3\nWater starting geometry\nO 0 0 0\nH 1.02 0 0\nH -0.28 0.98 0\n')
perception=analyze_structure(xyz,xyzin,report=root/'water.oracle.json')
coordinates=root/"water.coordinates.xyzin"
write_sonic_build_sections_from_cartesian(xyzin,coordinates,sycart=True)
result=optimize_native_sycart_matrix_gfn2(coordinates)
print('ORACLE atoms / point group:',perception.atom_count,perception.point_group)
print('LINK converged:',result['converged'])
print('Accepted steps:',result['iterations'])
print('Final energy / hartree:',result['energy'])
