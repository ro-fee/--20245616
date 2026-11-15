"""Export freshly trained checkpoints and their integrity manifest for inference."""
import argparse,json,shutil
from pathlib import Path
from predict import TAGS,sha
ROOT=Path(__file__).resolve().parent
WEIGHTS={'efficientnet_b0':.07,'convnext_tiny':.105,'convnext_base':.075,
         'dinov2_large':.25,'hsemotion':.25,'poster':.25}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'assets')
    args=p.parse_args();out=args.output
    if out.exists():raise ValueError('Choose a new asset directory')
    dino=ROOT/'experiments/dinov2_large_v1';special=ROOT/'experiments/fer_specialists_v1'
    base=json.loads((dino/'artifacts/base_manifest.json').read_text())
    if base['revision']!='47b73eefe95e8d44ec3623f8890bd894b6ea2d6c':raise ValueError('DINO revision mismatch')
    sources=[]
    for tag in TAGS:
        for fold in range(5):
            if tag in TAGS[:3]:source=ROOT/f'artifacts/{tag}_fold{fold}.pt'
            elif tag=='dinov2_large':source=dino/f'artifacts/lora_fold{fold}/best.pt'
            else:source=special/f'artifacts/{tag}_fold{fold}/final.pt'
            if not source.is_file():raise FileNotFoundError(source)
            sources.append((tag,fold,source))
    # Validate all sources before creating any export.
    supporting=base['weights']+base.get('config_files',[])
    if not any(r['name']=='config.json' for r in supporting):
        raise ValueError('Backbone config identity absent; rerun prepare with this source version')
    for rec in supporting:
        source=Path(base['snapshot'])/rec['name']
        if sha(source)!=rec['sha256']:raise ValueError('Modified generic backbone: '+rec['name'])
    out.mkdir(parents=True);(out/'weights').mkdir();checkpoints=[]
    for tag,fold,source in sources:
        dest=out/f'weights/{tag}_fold{fold}.pt';shutil.copy2(source,dest)
        checkpoints.append(dict(model=tag,fold=fold,path=dest.relative_to(out).as_posix(),sha256=sha(dest)))
    for rec in supporting:
        dest=out/'dinov2_base'/rec['name'];dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(base['snapshot'])/rec['name'],dest)
    shutil.copytree(special/'runtime_vendor/poster_models',out/'poster_models',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for folder in ('dinov2_base','poster_models')
        for p in sorted((out/folder).rglob('*')) if p.is_file()]
    manifest=dict(checkpoints=checkpoints,files=files,classes=['Angry','Fear','Happy','Sad','Surprise','Neutral'],weights=WEIGHTS)
    (out/'assets_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(out)
if __name__=='__main__':main()
