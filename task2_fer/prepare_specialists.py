"""Verify official pretraining checkpoints, then export tensor-only state dictionaries."""
import argparse,importlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hsemotion',type=Path,required=True);p.add_argument('--poster',type=Path,required=True)
    args=p.parse_args();exp=ROOT/'experiments/fer_specialists_v1'
    records=json.loads((exp/'download_manifest.json').read_text())
    sys.path.insert(0,str(exp/'src'));common=importlib.import_module('common')
    for tag in ('hsemotion','poster'):
        source=getattr(args,tag);rec=records[tag]
        if source.stat().st_size!=rec['size'] or common.sha(source)!=rec['sha256']:
            raise ValueError('Official pretraining checkpoint hash mismatch: '+tag)
    (exp/'weights').mkdir(exist_ok=True);(exp/'artifacts').mkdir(exist_ok=True)
    for tag in ('hsemotion','poster'):
        source=getattr(args,tag);dest=exp/records[tag]['relative_path'];dest.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve()!=dest.resolve():shutil.copy2(source,dest)
    importlib.import_module('models').convert()
if __name__=='__main__':main()
