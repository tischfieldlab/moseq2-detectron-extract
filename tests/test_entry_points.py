import glob
import os
import pkgutil
import subprocess
from importlib import import_module
from pathlib import Path
import sys

import pytest

__this_package = 'moseq2_detectron_extract'

# script_dir = Path(__file__).resolve().parent.parent.joinpath('moseq2_detectron_extract', 'scripts')
# scripts_to_test = [s for s in script_dir.glob('*.py') if s.name != '__init__.py']
# script_names = [s.name.replace('.py', '') for s in scripts_to_test]

# @pytest.mark.parametrize("entry_point", scripts_to_test, ids=script_names)
# def test_entry_point(entry_point):
#     rtn_code = subprocess.call(['python', str(entry_point), '--help'])
#     assert rtn_code == 0
print(os.environ.get("PATH"))


pkg_path = Path(__file__).resolve().parent.parent.joinpath(__this_package)
modules_to_test = pkgutil.iter_modules([pkg_path], prefix=__this_package + '.')
module_names = [m.name for m in modules_to_test]

@pytest.mark.parametrize("module_path", module_names)
def test_import(module_path):
    import_module(module_path, package=__this_package)
    assert True
