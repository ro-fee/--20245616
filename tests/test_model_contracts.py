from argparse import Namespace

import pytest
import torch

from conftest import load_module


@pytest.mark.parametrize('arch,encoder', [('unet','resnet34'),('unetpp','resnet34'),('unet','efficientnet-b3')])
def test_vessel_model_forward_without_downloads(vessel_common,arch,encoder):
    torch.set_num_threads(2)
    model=vessel_common.build_model(arch,encoder,False).eval()
    with torch.inference_mode():result=model(torch.randn(1,3,64,64))
    assert result.shape==(1,1,64,64) and torch.isfinite(result).all()


def test_fovea_checkpoint_roundtrip(tmp_path,fovea):
    torch.set_num_threads(2)
    cfg=fovea.OptimizedConfig(PRETRAINED=False,INPUT_SIZE=64,HEATMAP_SIZE=16)
    model=fovea.research_build_model(cfg,'coarse','cpu').eval()
    x=torch.randn(1,3,64,64)
    with torch.inference_mode():before=model(x)['logits']
    path=tmp_path/'model.pt'
    torch.save({'model':model.state_dict(),'use_visibility':False},path)
    other=fovea.research_build_model(cfg,'coarse','cpu')
    fovea.load_model_checkpoint(other,path,'cpu')
    with torch.inference_mode():after=other(x)['logits']
    torch.testing.assert_close(before,after,rtol=0,atol=0)


def test_fer_checkpoint_matches_inference_builder(tmp_path):
    from torchvision.models import efficientnet_b0
    torch.set_num_threads(2)
    infer=load_module('fer_checkpoint_test','task2_fer/predict.py')
    model=efficientnet_b0(weights=None);model.classifier[1]=torch.nn.Linear(1280,6);model.eval()
    (tmp_path/'weights').mkdir()
    torch.save(dict(state_dict=model.state_dict(),model='efficientnet_b0',fold=0),
        tmp_path/'weights/efficientnet_b0_fold0.pt')
    other=infer.build_model('efficientnet_b0',0,tmp_path,torch.float32).eval()
    x=torch.randn(1,3,160,160)
    with torch.inference_mode():before=model(x);after=other(x)
    torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-6)
    assert after.shape==(1,6)


def test_fer_lora_frozen_backbone_and_gradient_contract():
    torch.set_num_threads(2)
    dino=load_module('fer_dino_test','task2_fer/experiments/dinov2_large_v1/src/train_dino.py')
    cfg=dino.Dinov2Config(hidden_size=64,mlp_ratio=2,num_hidden_layers=2,
        num_attention_heads=4,image_size=28,patch_size=14)
    base=dino.Dinov2Model(cfg)
    args=Namespace(rank=4,alpha=8,lora_dropout=0.)
    model=dino.Classifier(base,args)
    x=torch.randn(2,3,28,28);target=torch.tensor([1,4])
    loss=torch.nn.functional.cross_entropy(model(x),target);loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for n,p in model.named_parameters() if 'lora_B' in n)


@pytest.mark.parametrize('tag,size',[('hsemotion',260),('poster',224)])
def test_specialist_heads_output_six_classes(tag,size):
    torch.set_num_threads(2)
    common=load_module('special_common_test','task2_fer/experiments/fer_specialists_v1/src/common.py')
    models=load_module('special_models_test','task2_fer/experiments/fer_specialists_v1/src/models.py',{'common':common})
    model=models.base_model(tag,classes=6).eval()
    with torch.inference_mode():result=model(torch.randn(1,3,size,size))
    assert result.shape==(1,6) and torch.isfinite(result).all()
