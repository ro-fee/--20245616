import ast
import csv
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import ROOT, load_module


def template(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)


def test_plant_training_splits_are_disjoint(tmp_path, monkeypatch):
    common = load_module('plant_common_test', 'task1_plants/common.py')
    source = ROOT/'task1_plants/original_train_manifest.json'
    (tmp_path/source.name).write_bytes(source.read_bytes())
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    records, splits = common.make_splits(json.loads((ROOT/'task1_plants/plan.json').read_text()))
    assert len(records) == 500 and len(splits['audit']) == 100
    seen = []
    for fold in splits['folds']:
        assert not set(fold['train']) & set(fold['validation'])
        assert not set(splits['audit']) & (set(fold['train']) | set(fold['validation']))
        seen += fold['validation']
    assert sorted(seen) == splits['development']


def test_fer_template_values_cannot_change_inputs(tmp_path):
    infer = load_module('fer_infer_test', 'task2_fer/predict.py')
    images = tmp_path/'images'
    images.mkdir()
    Image.fromarray(np.arange(48*48, dtype=np.uint8).reshape(48, 48)).save(images/'face.png')
    outputs = []
    for n, value in enumerate(['', 'invalid-target-text']):
        path = tmp_path/f'template{n}.csv'
        template(path, ['ID','Emotion'], [['face.png', value]])
        out = tmp_path/f'out{n}'
        out.mkdir()
        assert infer.prepare_pixels(images, path, out) == ['face.png']
        outputs.append(np.load(out/'test_pixels.npy'))
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize('bad_id', ['../face.png', '/face.png', 'sub/face.png', r'..\face.png'])
def test_fer_rejects_paths_in_template(tmp_path, bad_id):
    infer = load_module('fer_infer_path_test', 'task2_fer/predict.py')
    path = tmp_path/'template.csv'
    template(path, ['ID','Emotion'], [[bad_id, '']])
    with pytest.raises(ValueError):
        infer.prepare_pixels(tmp_path, path, tmp_path)


@pytest.mark.parametrize('filename',['81.jpg','0081.jpg'])
def test_fovea_template_is_label_independent(tmp_path, fovea_run,filename):
    Image.fromarray(np.zeros((60,80,3),np.uint8)).save(tmp_path/filename)
    frames = []
    for n, values in enumerate([['',''], ['100000','-9999']]):
        path = tmp_path/f'template{n}.csv'
        template(path,['ImageID','value'],list(zip(['81_Fovea_X','81_Fovea_Y'],values)))
        frame, _, layout = fovea_run.test_table(tmp_path,path)
        assert set(frame.columns) == {'image_id','path','width','height'}
        assert layout.axis.tolist() == ['x','y']
        frames.append(frame)
    pd.testing.assert_frame_equal(*frames)


def test_fovea_rejects_ambiguous_numeric_ids(tmp_path,fovea_run):
    for name in ['0081.jpg','00081.jpg']:
        Image.fromarray(np.zeros((60,80,3),np.uint8)).save(tmp_path/name)
    path=tmp_path/'template.csv'
    template(path,['ImageID','value'],[['81_Fovea_X',''],['81_Fovea_Y','']])
    with pytest.raises(ValueError):fovea_run.test_table(tmp_path,path)


def test_fovea_rejects_multiple_ids_for_one_image(tmp_path,fovea_run):
    Image.fromarray(np.zeros((60,80,3),np.uint8)).save(tmp_path/'0081.jpg')
    path=tmp_path/'template.csv'
    template(path,['ImageID','value'],[[i,''] for i in ['81_Fovea_X','81_Fovea_Y','0081_Fovea_X','0081_Fovea_Y']])
    with pytest.raises(ValueError):fovea_run.test_table(tmp_path,path)


def test_fovea_rejects_incomplete_axis_pairs(tmp_path, fovea_run):
    path=tmp_path/'template.csv'
    template(path,['ImageID','value'],[['81_Fovea_X','']])
    with pytest.raises(ValueError):
        fovea_run.test_table(tmp_path,path)


def test_fovea_duplicate_groups_do_not_cross_folds(fovea):
    # Pixel duplicates override different patient identifiers.
    frame=pd.DataFrame(dict(image_id=[str(i) for i in range(20)],
        patient_group=[str(i) for i in range(20)],
        pixel_hash=['same' if i<2 else str(i) for i in range(20)],visible=True))
    frame['group']=fovea.connected_groups(frame)
    frame=fovea.assign_folds(frame,fovea.OptimizedConfig())
    assert frame.groupby('group').fold.nunique().max()==1
    assert frame.iloc[0].fold==frame.iloc[1].fold


def test_training_fer_channel_transform_is_spawn_picklable():
    module=load_module('fer_cnn_test','task2_fer/src/train_large.py')
    transform=pickle.loads(pickle.dumps(module.make_transform(False)))
    assert transform(Image.fromarray(np.zeros((48,48),np.uint8))).shape==(3,160,160)


def test_fer_preprocessing_has_no_test_directory_read():
    source=(ROOT/'task2_fer/src/audit.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    assert not any(isinstance(n,ast.Constant) and isinstance(n.value,str) and
                   ('test_pixels' in n.value or n.value=='test') for n in ast.walk(tree))


def test_python_sources_compile_and_notebook_has_no_outputs():
    for path in ROOT.rglob('*.py'):
        compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
    notebook=json.loads((ROOT/'task3_fovea/fovea.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            assert cell['outputs']==[] and cell['execution_count'] is None
