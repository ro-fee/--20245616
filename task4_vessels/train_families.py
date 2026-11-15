"""Run six fixed families with separate cross-validation and full-training runs."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FAMILIES=json.loads((ROOT/'families.json').read_text())

def commands(args):
    modes=['cv','full'] if args.mode=='both' else [args.mode]
    for name,cfg in FAMILIES.items():
        if args.family!='all' and args.family!=name:continue
        for mode in modes:
            command=[sys.executable,str(ROOT/'train.py'),'--data',str(args.data.resolve()),
                '--out',str((args.runs_root/name/mode).resolve()),'--workers',str(args.workers)]
            if mode=='full':command+=['--full']
            for key,value in cfg.items():
                if key=='init':command+=['--init_run',str((args.runs_root/value/mode).resolve())]
                elif isinstance(value,bool):
                    if value:command+=['--'+key]
                else:command+=['--'+key,str(value)]
            if args.encoder_weights:command+=['--encoder-weights',str(args.encoder_weights.resolve())]
            yield command

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--runs-root',type=Path,default=ROOT/'runs')
    p.add_argument('--family',choices=['all',*FAMILIES],default='all')
    p.add_argument('--mode',choices=['cv','full','both'],default='both')
    p.add_argument('--encoder-weights',type=Path)
    p.add_argument('--workers',type=int,default=0);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    for command in commands(args):
        print(json.dumps(command,ensure_ascii=False),flush=True)
        if not args.dry_run:subprocess.run(command,check=True,cwd=ROOT)
if __name__=='__main__':main()
