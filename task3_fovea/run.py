"""Train the optimized coarse/fine pipeline, or predict from saved folds."""
import argparse
import json
from dataclasses import fields
from pathlib import Path
import pandas as pd
import torch
import engine as e

def test_table(test_dir,template):
    sample=pd.read_csv(template,dtype=str,keep_default_na=False)
    if list(sample.columns)!=['ImageID','value'] or sample.ImageID.duplicated().any():
        raise ValueError('Expected unique ImageID,value template')
    layout=pd.DataFrame([e.parse_submission_id(v) for v in sample.ImageID],columns=['image_id','axis'])
    counts=layout.groupby(['image_id','axis']).size().unstack(fill_value=0)
    if set(counts.columns)!={'x','y'} or not (counts[['x','y']]==1).all().all():
        raise ValueError('Every image requires one X and one Y row')
    root=Path(test_dir).resolve()
    images=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in e.IMAGE_EXTENSIONS]
    index={}
    for path in images:
        if not path.resolve().is_relative_to(root):raise ValueError('Image escaped input directory')
        index.setdefault(e.canonical_image_id(path.name),[]).append(path)
    rows=[]
    for image_id in dict.fromkeys(layout.image_id):
        choices=index.get(image_id,[])
        if len(choices)!=1:raise ValueError('Image identifier is absent or ambiguous: '+image_id)
        path=choices[0];im=e.read_rgb(path);h,w=im.shape[:2]
        rows.append(dict(image_id=image_id,path=str(path),width=w,height=h))
    if not rows:raise ValueError('Template is empty')
    return pd.DataFrame(rows),sample,layout

def predict(args):
    recorded=json.loads((args.models/'run_config.json').read_text())['config']
    allowed={f.name for f in fields(e.OptimizedConfig)}
    cfg=e.OptimizedConfig(**{k:v for k,v in recorded.items() if k in allowed})
    cfg.PRETRAINED=False;cfg.ALLOW_WEIGHT_DOWNLOAD=False;cfg.NUM_WORKERS=0
    cfg.SUBMISSION_PATH=str(args.output);cfg.AMP=args.device.startswith('cuda')
    test,sample,layout=test_table(args.test_dir,args.template)
    predictions=[];device=torch.device(args.device)
    for fold in range(5):
        folder=args.models/f'fold_{fold}'
        coarse=e.research_build_model(cfg,'coarse',device)
        e.load_model_checkpoint(coarse,folder/'coarse_final/model.pt',device)
        cp=e.predict_research(coarse,test,cfg,device,'coarse')
        del coarse
        fine=e.research_build_model(cfg,'fine',device)
        e.load_model_checkpoint(fine,folder/'fine_final/model.pt',device)
        _,bp,_=e.optimized_inference(fine,test,cp,cfg,device);predictions.append(bp)
        del fine
        if device.type=='cuda':torch.cuda.empty_cache()
    mean=pd.concat(predictions).groupby('image_id',as_index=False)[['x','y']].mean()
    result=test.merge(mean,on='image_id',validate='one_to_one');result['visibility_prob']=1.
    e.write_submission(sample,layout,result,{'id_column':'ImageID','value_column':'value'},cfg,.5)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    tr=sub.add_parser('train');tr.add_argument('--data',type=Path,required=True)
    tr.add_argument('--labels',type=Path,required=True,help='Training coordinate CSV only')
    tr.add_argument('--template',type=Path,required=True);tr.add_argument('--pretrained',type=Path,required=True)
    tr.add_argument('--output-dir',type=Path,required=True);tr.add_argument('--workers',type=int,default=0)
    pr=sub.add_parser('predict');pr.add_argument('--test-dir',type=Path,required=True)
    pr.add_argument('--template',type=Path,required=True);pr.add_argument('--models',type=Path,required=True)
    pr.add_argument('--output',type=Path,required=True);pr.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.mode=='predict':return predict(args)
    out=args.output_dir.resolve()
    cfg=e.OptimizedConfig(DATA_DIR=str(args.data.resolve()),LABEL_CSV=str(args.labels.resolve()),
        SAMPLE_CSV=str(args.template.resolve()),PRETRAINED_WEIGHTS_PATH=str(args.pretrained.resolve()),
        OUTPUT_DIR=str(out),SUBMISSION_PATH=str(out/'submission.csv'),NUM_WORKERS=args.workers,
        ALLOW_WEIGHT_DOWNLOAD=False)
    e.run_optimized(cfg)
if __name__=='__main__':main()
