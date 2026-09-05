## @package tests.test_super_resolved_func
#  Regressions for conservative super method display paths.

from pathlib import Path

import pytest

from pcresolve import analyze_project


FIXTURE = Path(__file__).parent / 'fixtures' / 'super_resolved_func'


@pytest.mark.parametrize('module', [
    'direct', 'alias', 'module_alias', 'keras_registered',
])
def test_external_super_path(module):
    calls = analyze_project(str(FIXTURE / module)).all_api_calls
    for method, arguments in [('__init__', '**kwargs'), ('get_config', '')]:
        name = 'super().' + method
        call = next(c for c in calls if c.func_name == name)
        expected = 'tensorflow.keras.layers.Layer.' + method
        assert call.top_library == 'tensorflow'
        assert call.expression == name + '(' + arguments + ')'
        assert call.func_name == name
        assert call.resolved_func == expected
        assert call.resolved_chain == [name, expected, 'tensorflow']


@pytest.mark.parametrize('module', [
    'multiple', 'explicit', 'dynamic', 'subscript', 'local',
    'shadowed', 'decorated', 'metaclass',
    'super_shadowed', 'same_library_multiple', 'nested_function',
])
def test_ambiguous_super_path_stays_original(module):
    calls = analyze_project(str(FIXTURE / module)).all_api_calls
    call = next(c for c in calls if c.func_name.endswith('.get_config'))
    assert call.resolved_func == call.func_name


def test_base_path_snapshot_survives_later_rebinding():
    calls = analyze_project(str(FIXTURE / 'rebound')).all_api_calls
    for call in calls:
        if call.func_name.startswith('super().'):
            assert call.resolved_func == (
                'tensorflow.keras.layers.Layer.' + call.func_name.split('.')[-1])


def test_same_named_classes_keep_separate_base_paths():
    calls = analyze_project(str(FIXTURE / 'same_name')).all_api_calls
    paths = [c.resolved_func for c in calls if c.func_name == 'super().get_config']
    assert paths == ['tensorflow.keras.layers.Layer.get_config',
                     'otherlib.Parent.get_config']


@pytest.mark.parametrize('import_line, decorator', [
    ('from tensorflow.keras.utils import register_keras_serializable as register',
     '@register(package="MyLayers")'),
    ('from tensorflow.keras.utils import register_keras_serializable as register',
     '@register("MyLayers", name="CustomLayer")'),
    ('from tensorflow.keras.utils import register_keras_serializable as register',
     '@register(name=None)'),
])
def test_registered_decorator_import_aliases(tmp_path, import_line, decorator):
    source = (FIXTURE / 'keras_registered' / 'example.py').read_text()
    source = source.replace(
        'from tensorflow.keras.utils import register_keras_serializable', import_line)
    source = source.replace(
        '@register_keras_serializable(package="MyLayers")', decorator)
    (tmp_path / 'example.py').write_text(source)
    calls = analyze_project(str(tmp_path)).all_api_calls
    paths = {c.func_name: c.resolved_func for c in calls}
    for method in ('__init__', 'get_config'):
        assert paths['super().' + method] == 'tensorflow.keras.layers.Layer.' + method


@pytest.mark.parametrize('setup, decorators', [
    ('', '@unknown\n@register_keras_serializable(package="MyLayers")'),
    ('', '@register_keras_serializable(package="MyLayers")\n@unknown'),
    ('def register_keras_serializable(**kwargs):\n    return unknown',
     '@register_keras_serializable(package="MyLayers")'),
    ('from otherlib import register_keras_serializable',
     '@register_keras_serializable(package="MyLayers")'),
    ('register_keras_serializable = unknown',
     '@register_keras_serializable(package="MyLayers")'),
    ('if condition:\n    register_keras_serializable = unknown',
     '@register_keras_serializable(package="MyLayers")'),
    ('from otherlib import *',
     '@register_keras_serializable(package="MyLayers")'),
    ('(register_keras_serializable := unknown)',
     '@register_keras_serializable(package="MyLayers")'),
    ('register_keras_serializable += unknown',
     '@register_keras_serializable(package="MyLayers")'),
    ('', '@tf.keras.utils.register_keras_serializable(package="MyLayers")'),
    ('from tensorflow.keras import utils\nutils.register_keras_serializable = unknown',
     '@utils.register_keras_serializable(package="MyLayers")'),
    ('', '@register_keras_serializable'),
    ('', '@register_keras_serializable(**options)'),
    ('', '@register_keras_serializable(package=make_package())'),
])
def test_unproven_registration_decorator_stays_original(tmp_path, setup, decorators):
    source = (FIXTURE / 'keras_registered' / 'example.py').read_text()
    source = source.replace(
        '@register_keras_serializable(package="MyLayers")', setup + '\n' + decorators)
    (tmp_path / 'example.py').write_text(source)
    calls = analyze_project(str(tmp_path)).all_api_calls
    super_calls = [c for c in calls if c.func_name.startswith('super().')]
    assert len(super_calls) == 2
    assert all(c.resolved_func == c.func_name for c in super_calls)


@pytest.mark.parametrize('local_path', ['tensorflow.py', 'tensorflow/keras/utils.py'])
def test_local_tensorflow_decorator_does_not_gain_external_contract(tmp_path, local_path):
    local_module = tmp_path / local_path
    local_module.parent.mkdir(parents=True, exist_ok=True)
    local_module.write_text('keras = unknown\n')
    (tmp_path / 'example.py').write_text('''\
from tensorflow.keras.utils import register_keras_serializable
from json import JSONDecoder
@register_keras_serializable(package="MyLayers")
class CustomDecoder(JSONDecoder):
    def decode(self, source):
        return super().decode(source)
''')
    calls = analyze_project(str(tmp_path)).all_api_calls
    call = next(c for c in calls if c.func_name == 'super().decode')
    assert call.resolved_func == call.func_name


def test_function_local_conditional_registration_remains_unresolved(tmp_path):
    (tmp_path / 'example.py').write_text('''\
from tensorflow.keras.layers import Layer
def build_class():
    try:
        from otherlib import register_keras_serializable
    except ImportError:
        from tensorflow.keras.utils import register_keras_serializable
    @register_keras_serializable(package="MyLayers")
    class CustomLayer(Layer):
        def get_config(self):
            return super().get_config()
    return CustomLayer
''')
    calls = analyze_project(str(tmp_path)).all_api_calls
    call = next(c for c in calls if c.func_name == 'super().get_config')
    assert call.resolved_func == call.func_name
