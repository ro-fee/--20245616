import hashlib,json,random,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score,recall_score

EXP=Path(__file__).resolve().parents[1]
ROOT=EXP.parents[1]
GIB=1024**3
CFG=json.loads((EXP/'config.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def emit(event,**data):
 print(json.dumps(dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=event,**data)),flush=True)
def metric(prob,y):
 pred=prob.argmax(1)
 return dict(accuracy=float((pred==y).mean()),correct=int((pred==y).sum()),count=len(y),
  macro_f1=float(f1_score(y,pred,labels=list(range(6)),average='macro',zero_division=0)),
  recall=recall_score(y,pred,labels=list(range(6)),average=None,zero_division=0).tolist())
def probability(p,n):
 assert p.shape==(n,6) and np.isfinite(p).all() and (p>=0).all()
 assert np.allclose(p.sum(1),1,atol=1e-5)
def seed(value):
 random.seed(value);np.random.seed(value);torch.manual_seed(value)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
def data(fold):
 df=pd.read_csv(ROOT/'artifacts/train_manifest.csv')
 assert len(df)==28275 and np.array_equal(df['index'],np.arange(len(df)))
 assert np.array_equal(df.emotion.map(CFG['classes']),df.label)
 assert set(df.fold)==set(range(5)) and df.groupby('group').fold.nunique().max()==1
 tr=np.flatnonzero(df.fold.to_numpy()!=fold);va=np.flatnonzero(df.fold.to_numpy()==fold)
 assert not(set(df.iloc[tr].group)&set(df.iloc[va].group))
 pixels=np.load(ROOT/'data/train_pixels.npy',mmap_mode='r')
 assert pixels.shape==(28275,48,48)
 return df,tr,va,pixels
def disk(extra=0):
 free=shutil.disk_usage(EXP).free/GIB
 if free<CFG['disk_reserve_gib']+extra:raise RuntimeError(f'DISK_RESERVE: free={free:.3f} GiB')
def identity(model,fold,stop_epoch):
 import timm
 return dict(model=model,fold=fold,stop_epoch=stop_epoch,config_sha256=sha(EXP/'config.json'),
  code={p.name:sha(p) for p in sorted((EXP/'src').glob('*.py'))},
  runtime_code={str(p.relative_to(EXP)):sha(p) for p in sorted((EXP/'runtime_vendor').rglob('*.py'))},
  download_manifest_sha256=sha(EXP/'download_manifest.json'),timm=timm.__version__,
  manifest_sha256=sha(ROOT/'artifacts/train_manifest.csv'),
  base_sha256=sha(EXP/f'weights/{model}_state.pt'),torch=str(torch.__version__))

def locked_write(path,data):
 path=Path(path)
 if path.exists():
  assert json.loads(path.read_text())==data,'Frozen receipt differs: '+str(path)
 else:write(path,data)
