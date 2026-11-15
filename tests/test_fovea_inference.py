import numpy as np
import pandas as pd
import torch
from PIL import Image


def test_prediction_whitelist_removes_all_target_fields(tmp_path,fovea,monkeypatch):
    path=tmp_path/'81.png'
    Image.fromarray(np.zeros((60,80,3),np.uint8)).save(path)
    frame=pd.DataFrame([dict(image_id='81',path=str(path),width=80,height=60,
        x=-100000,y=999999,visible=False)])
    cfg=fovea.OptimizedConfig(INPUT_SIZE=64,HEATMAP_SIZE=16,NUM_WORKERS=0,AMP=False,TTA=False)
    captured=[]
    original=fovea.make_research_loader
    def loader(frame,*args,**kwargs):
        captured.append(frame.copy())
        return original(frame,*args,**kwargs)
    monkeypatch.setattr(fovea,'make_research_loader',loader)
    class Heatmap(torch.nn.Module):
        def forward(self,x):
            batch=len(x)
            logits=torch.zeros(batch,1,16,16,device=x.device);logits[:,:,8,8]=8
            return {'logits':logits}
    # Use the real dataset/coordinate/TTA decoder with a small deterministic head.
    result=fovea.predict_research(Heatmap(),frame,cfg,'cpu')
    frame[['x','y']]=[1.,2.];frame['visible']=True
    other=fovea.predict_research(Heatmap(),frame,cfg,'cpu')
    assert set(captured[0].columns)=={'image_id','path','width','height'}
    np.testing.assert_allclose(result[['x','y']],other[['x','y']],rtol=0,atol=0)
    assert 0<=result.x.iloc[0]<80 and 0<=result.y.iloc[0]<60


def test_fine_dataset_requires_prediction_crop(tmp_path,fovea):
    import pytest
    frame=pd.DataFrame([dict(image_id='81',path='unused',width=80,height=60)])
    with pytest.raises(ValueError,match='prediction-derived'):
        fovea.ResearchDataset(frame,fovea.OptimizedConfig(),False,'fine')
