import os
import pkgutil
import subprocess
from importlib import import_module
from pathlib import Path

import pytest
from moseq2_detectron_extract.cli import cli


__THIS_PACKAGE = 'moseq2_detectron_extract'
pkg_path = Path(__file__).resolve().parent.parent.joinpath(__THIS_PACKAGE)


@pytest.mark.parametrize("entry_point", [value.name for value in cli.commands.values()])
def test_entry_point(entry_point):
    ''' Test that we can run commands with the --help flag
    '''
    rtn_code = subprocess.call(['python', os.path.join(pkg_path, 'cli.py'), str(entry_point), '--help'])
    assert rtn_code == 0


modules_to_test = pkgutil.iter_modules([pkg_path], prefix=__THIS_PACKAGE + '.')
@pytest.mark.parametrize("module_path", [m.name for m in modules_to_test])
def test_import(module_path):
    ''' Test that we can import all modules in the package
    '''
    import_module(module_path, package=__THIS_PACKAGE)
    assert True
