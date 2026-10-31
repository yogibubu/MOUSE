from remote_bootstrap import HOST,BASE
import pytest
raise SystemExit(pytest.main([str(BASE/'source/qualification'/x) for x in ['test_deltascf.py','test_roks.py','test_deltascf_link.py','test_orbital_saddle.py']]+['-q']))
