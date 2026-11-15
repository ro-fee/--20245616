"""Unlabeled-image inference: six families, five folds plus a full-data model each."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common import build_model, read_image, preprocess, predict_prob
from protocol import fuse, rle_encode
ROOT=Path(__file__).resolve().parent

def template_ids(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['Id','Predicted']:
            raise ValueError("Expected template columns Id,Predicted")
        ids=[r['Id'] for r in reader]
    if not ids or len(ids)!=len(set(ids)) or any(not i.isdigit() for i in ids):
        raise ValueError('IDs must be unique numeric image identifiers')
    return ids

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-dir',type=Path,required=True);p.add_argument('--template',type=Path,required=True)
    p.add_argument('--runs-root',type=Path,default=ROOT/'runs');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    ids=template_ids(args.template)
    raw=[read_image(args.test_dir/(i+'.jpg')) for i in ids]
    cfgs=json.loads((ROOT/'families.json').read_text());averages=[[] for _ in ids]
    for family,cfg in cfgs.items():
        images=[np.array(Image.fromarray(im).resize((512,512),Image.Resampling.BILINEAR))
            if cfg.get('target512') else im for im in raw]
        images=[preprocess(im,1) for im in images]
        sums=[np.zeros(im.shape[:2],np.float32) for im in images]
        for mode,tags in [('cv',[f'fold{k}' for k in range(5)]),('full',['full'])]:
            folder=args.runs_root/family/mode
            saved=json.loads((folder/'args.json').read_text())
            for key in ('arch','encoder','scale'):
                if saved[key]!=cfg[key]:raise ValueError('Model configuration mismatch: '+family)
            if saved['full']!=(mode=='full'):raise ValueError('Fold/full model mismatch')
            for tag in tags:
                model=build_model(cfg['arch'],cfg['encoder'],False).cuda().eval()
                model.load_state_dict(torch.load(folder/(tag+'.pt'),map_location='cpu',weights_only=True),strict=True)
                for j,im in enumerate(images):
                    a=predict_prob(model,im,cfg['scale'])
                    b=predict_prob(model,np.ascontiguousarray(im.transpose(1,0,2)),cfg['scale']).T
                    sums[j]+=(a+b)/12
                del model;torch.cuda.empty_cache()
        for j,prob in enumerate(sums):averages[j].append(prob)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['Id','Predicted'])
        w.writerows((i,rle_encode(fuse(prob))) for i,prob in zip(ids,averages))
    print(args.output)
if __name__=='__main__':main()
