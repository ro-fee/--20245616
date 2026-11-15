import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative, aliases=None):
    aliases = aliases or {}
    saved = {key: sys.modules.get(key) for key in aliases}
    sys.modules.update(aliases)
    path = ROOT / relative
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


@pytest.fixture(scope='session')
def fovea():
    return load_module('fovea_engine_test', 'task3_fovea/engine.py')


@pytest.fixture(scope='session')
def fovea_run(fovea):
    return load_module('fovea_run_test', 'task3_fovea/run.py', {'engine': fovea})


@pytest.fixture(scope='session')
def vessel_protocol():
    return load_module('vessel_protocol_test', 'task4_vessels/protocol.py')


@pytest.fixture(scope='session')
def vessel_common():
    return load_module('vessel_common_test', 'task4_vessels/common.py')


@pytest.fixture(scope='session')
def vessel_train(vessel_protocol, vessel_common):
    return load_module('vessel_train_test', 'task4_vessels/train.py',
                       {'protocol': vessel_protocol, 'common': vessel_common})
