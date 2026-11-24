"""Load byte-identical PDD modules in a private namespace, without sys.path edits.

The old code uses absolute local imports and source hashes. A per-module import
hook resolves those names privately; unrelated code keeps normal Python imports.
No legacy module names are installed in sys.modules and no code is rewritten.
"""
import builtins
import importlib.util
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType

_ROOT = Path(__file__).with_name('_pdd_frozen')
_PREFIX = '_stage4_pdd_frozen_v1'
_LOCK = RLock()
_FILES = {
    'agent4_pdd_common_v2': 'agent4_pdd_common_v2.py',
    'build_agent4_pdd_pilot': 'build_agent4_pdd_pilot.py',
    'train_pdd_baseline': 'train_pdd_baseline.py',
    'audit_compatibility': 'agent4_integration_audit/audit_compatibility.py',
}


def _private_import(name, globals=None, locals=None, fromlist=(), level=0):
    if not level:
        if name in _FILES:
            return _load(name)
        if name == 'agent4_integration_audit.audit_compatibility':
            return _load('audit_compatibility')
        if name == 'agent4_integration_audit':
            package_name = _PREFIX + '.agent4_integration_audit'
            package = sys.modules.get(package_name)
            if package is None:
                package = ModuleType(package_name)
                package.__path__ = [str(_ROOT/'agent4_integration_audit')]
                sys.modules[package_name] = package
            package.audit_compatibility = _load('audit_compatibility')
            return package
    return builtins.__import__(name, globals, locals, fromlist, level)


def _load(name):
    with _LOCK:
        qualified = _PREFIX + '.' + name
        if qualified in sys.modules:
            return sys.modules[qualified]
        if _PREFIX not in sys.modules:
            package = ModuleType(_PREFIX)
            package.__path__ = [str(_ROOT)]
            sys.modules[_PREFIX] = package
        spec = importlib.util.spec_from_file_location(qualified, _ROOT/_FILES[name])
        module = importlib.util.module_from_spec(spec)
        module.__dict__['__builtins__'] = {**vars(builtins), '__import__': _private_import}
        sys.modules[qualified] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(qualified, None)
            raise
        return module


def load_frozen_runtime():
    return _load('audit_compatibility')
