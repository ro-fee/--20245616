import copy

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import load_module


@pytest.mark.parametrize('kind',['empty','full','corners','random'])
def test_rle_roundtrip_and_row_order(vessel_protocol,kind):
    mask=np.zeros((7,11),bool)
    if kind=='full':mask[:]=True
    elif kind=='corners':mask[0,0]=mask[-1,-1]=True
    elif kind=='random':mask=np.random.default_rng(9).random(mask.shape)>.7
    encoded=vessel_protocol.rle_encode(mask)
    np.testing.assert_array_equal(vessel_protocol.rle_decode(encoded,mask.shape),mask)
    if kind=='corners':assert encoded=='1 1 77 1'


@pytest.mark.parametrize('bad',['0 1','1 0','1','1 3 2 4','77 2'])
def test_rle_rejects_invalid_intervals(vessel_protocol,bad):
    with pytest.raises(ValueError):vessel_protocol.rle_decode(bad,(7,11))


def test_equal_family_fusion_is_fixed(vessel_protocol):
    assert not vessel_protocol.fuse([np.full((12,13),.5,np.float32)]*6).any()
    assert vessel_protocol.fuse([np.full((12,13),.51,np.float32)]*6).all()
    with pytest.raises(ValueError):vessel_protocol.fuse([np.ones((12,13))])
    with pytest.raises(ValueError):vessel_protocol.fuse([np.full((2,2),np.nan)]*6)


def test_warm_start_rejects_full_model_and_other_data(vessel_protocol):
    current=dict(encoder='resnet34',arch='unet',folds=5,full=False,
        training_fingerprint='abc',validation_ids=[['a'],['b']],encoder_only_init=False)
    vessel_protocol.check_warm_start(current,current)
    for key,value in [('full',True),('training_fingerprint','other'),('validation_ids',[['b'],['a']])]:
        old=copy.deepcopy(current);old[key]=value
        with pytest.raises(ValueError):vessel_protocol.check_warm_start(current,old)


def test_mask_changes_modify_training_identity(tmp_path,vessel_protocol):
    for role in ('image','label'):
        (tmp_path/role).mkdir();(tmp_path/role/'1.jpg').write_bytes(role.encode())
    before=vessel_protocol.training_fingerprint(tmp_path,['1'])
    (tmp_path/'label/1.jpg').write_bytes(b'other-mask')
    assert vessel_protocol.training_fingerprint(tmp_path,['1'])!=before


def test_image_decode_uses_content_not_extension(tmp_path,vessel_common):
    array=np.zeros((41,53,3),np.uint8);array[:,:,0]=180
    path=tmp_path/'1.jpg';Image.fromarray(array).save(path,format='TIFF')
    np.testing.assert_array_equal(vessel_common.read_image(path),array)


def test_tta_returns_original_geometry(vessel_common):
    class PixelModel(torch.nn.Module):
        def forward(self,x):return x[:,:1]*.1
    image=np.random.default_rng(8).integers(0,255,(65,81,3),dtype=np.uint8)
    plain=vessel_common.predict_prob(PixelModel(),image,1.,False,'cpu')
    tta=vessel_common.predict_prob(PixelModel(),image,1.,True,'cpu')
    assert tta.shape==(65,81) and np.isfinite(tta).all()
    np.testing.assert_allclose(plain,tta,atol=1e-7)


def test_segmentation_loss_has_finite_gradients(vessel_train):
    logits=torch.randn(2,1,32,32,requires_grad=True)
    y=(torch.rand(2,1,32,32)>.5).float()
    loss=vessel_train.loss_fn(logits,y);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_family_plan_separates_cv_and_full_initializers(tmp_path):
    from argparse import Namespace
    planner=load_module('vessel_plan_test','task4_vessels/train_families.py')
    args=Namespace(mode='both',family='all',data=tmp_path/'absent',runs_root=tmp_path/'runs',workers=0,encoder_weights=None)
    commands=list(planner.commands(args))
    assert len(commands)==12 and not args.runs_root.exists()
    for command in commands:
        if '--init_run' in command:
            dest=command[command.index('--out')+1]
            source=command[command.index('--init_run')+1]
            assert dest.endswith('full')==source.endswith('full')
            assert ('--full' in command)==dest.endswith('full')
