import importlib.util
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
import pytest
raise SystemExit(pytest.main([str(BASE/'source/qualification/test_roks_geometry_only.py')+'::test_positive_density_transport_has_analytic_second_metric_and_frame_jets','-q']))
