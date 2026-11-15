"""Evaluate the fixed ensemble on grouped training folds; accepts no test input."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from export_assets import WEIGHTS
ROOT=Path(__file__).resolve().parent

def main():
    manifest=pd.read_csv(ROOT/'artifacts/train_manifest.csv');y=manifest.label.to_numpy()
    if manifest.groupby('group').fold.nunique().max()!=1:raise ValueError('A duplicate group crosses folds')
    total=np.zeros((len(y),6),np.float64)
    for tag,weight in WEIGHTS.items():
        oof=np.zeros_like(total);covered=np.zeros(len(y),bool)
        for fold in range(5):
            if tag in ('efficientnet_b0','convnext_tiny','convnext_base'):
                path=ROOT/f'artifacts/{tag}_fold{fold}_preds.npz'
            elif tag=='dinov2_large':path=ROOT/f'experiments/dinov2_large_v1/artifacts/lora_fold{fold}/preds.npz'
            else:path=ROOT/f'experiments/fer_specialists_v1/artifacts/{tag}_fold{fold}/final.npz'
            with np.load(path,allow_pickle=False) as data:
                idx=data['valid_idx'];expected=np.flatnonzero(manifest.fold.to_numpy()==fold)
                if not np.array_equal(idx,expected):raise ValueError('Fold identity mismatch')
                plain=data['plain'] if tag in ('poster','hsemotion') else data['val_orig']
                if tag in ('efficientnet_b0','convnext_tiny','poster'):
                    plain=data['flip'] if tag=='poster' else (plain+data['val_flip'])/2
                if plain.shape!=(len(idx),6) or not np.isfinite(plain).all() or not np.allclose(plain.sum(1),1,atol=1e-5):
                    raise ValueError('Invalid fold probabilities')
                oof[idx]=plain;covered[idx]=True
        if not covered.all():raise ValueError('Incomplete OOF coverage')
        total+=weight*oof
    pred=total.argmax(1)
    print(json.dumps(dict(basis='training_grouped_oof',rows=len(y),accuracy=accuracy_score(y,pred),
        macro_f1=f1_score(y,pred,average='macro')),indent=2))
if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
