from remote_bootstrap import HOST,BASE
import pytest
files=['test_deltascf.py','test_roks.py','test_deltascf_link.py','test_orbital_saddle.py','test_hybrid_spectrum.py','test_energy_only_checkpoint.py']
raise SystemExit(pytest.main([str(BASE/'source/qualification'/f) for f in files]+['-q']))
