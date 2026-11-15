"""Official models, strictly restored before replacing their classification heads."""
import sys,zipfile,pickletools
from pathlib import Path
import torch,timm
from torch import nn
from common import EXP,sha,write

def head(model,tag):return model.classifier if tag=='hsemotion' else model.VIT.head
def new_head(model,tag):
 if tag=='hsemotion':model.classifier=nn.Linear(model.classifier.in_features,6)
 else:
  # Author's ClassificationHead is a Sequential containing LayerNorm then Linear.
  old=model.VIT.head
  for name,layer in list(old.named_modules()):
   if isinstance(layer,nn.Linear) and layer.out_features==7:
    parent=old
    parts=name.split('.')
    for part in parts[:-1]:parent=getattr(parent,part)
    setattr(parent,parts[-1],nn.Linear(layer.in_features,6));break
  else:raise RuntimeError('POSTER output head not found')
 return model
def poster_base():
 sys.path.insert(0,str(EXP/'runtime_vendor'))
 from poster_models.PosterV2_7cls import pyramid_trans_expr2
 return pyramid_trans_expr2(img_size=224,num_classes=7)
def base_model(tag,classes=7):
 m=timm.create_model('tf_efficientnet_b2',pretrained=False,num_classes=7,global_pool='max') if tag=='hsemotion' else poster_base()
 if classes==6:new_head(m,tag)
 return m
class RecorderMeter:pass
class RecorderMeter1:pass
def official_load(path,tag):
 # Original official checkpoints contain either a complete timm Module or inert plot recorders.
 # Verify the pinned download manifest and reject unexpected pickle globals before loading.
 manifest=__import__('json').loads((EXP/'download_manifest.json').read_text())
 rec=manifest[tag];assert sha(path)==rec['sha256'] and Path(path).stat().st_size==rec['size']
 with zipfile.ZipFile(path) as z:
  name=next(n for n in z.namelist() if n.endswith('data.pkl'))
  globals_=sorted({arg for op,arg,_ in pickletools.genops(z.read(name)) if op.name=='GLOBAL'})
 allowed=('torch ','torch._utils ','torch.nn.modules.','timm.layers.','timm.models.','collections OrderedDict',
          '__builtin__ set','_codecs encode','numpy dtype','numpy ndarray','numpy.core.multiarray _reconstruct',
          '__main__ RecorderMeter','__main__ RecorderMeter1')
 assert all(g.startswith(allowed) for g in globals_),globals_
 setattr(sys.modules['__main__'],'RecorderMeter',RecorderMeter)
 setattr(sys.modules['__main__'],'RecorderMeter1',RecorderMeter1)
 return torch.load(path,map_location='cpu',weights_only=False),globals_
def convert():
 records={}
 for tag,name in [('hsemotion','vendor/hsemotion/models/affectnet_emotions/enet_b2_7.pt'),('poster','weights/poster_affectnet7.pth')]:
  target=EXP/f'weights/{tag}_state.pt'
  if target.exists():
   r=__import__('json').loads((EXP/f'artifacts/{tag}_conversion.json').read_text());assert sha(target)==r['state_sha256'];continue
  original,globals_=official_load(EXP/name,tag)
  state=original.state_dict() if isinstance(original,nn.Module) else original['state_dict']
  state={k.removeprefix('module.'):v.detach().cpu() for k,v in state.items()}
  m=base_model(tag);m.load_state_dict(state,strict=True)
  torch.save(state,target)
  records[tag]=dict(state_sha256=sha(target),state_size=target.stat().st_size,strict_full_restore=True,
                   pickle_globals=globals_,source_checkpoint_sha256=sha(EXP/name),parameters=sum(p.numel() for p in m.parameters()))
  write(EXP/f'artifacts/{tag}_conversion.json',records[tag])
 return records
def model(tag):
 m=base_model(tag);m.load_state_dict(torch.load(EXP/f'weights/{tag}_state.pt',map_location='cpu',weights_only=True),strict=True)
 return new_head(m,tag)
def mode(m,tag,head_only=False):
 m.eval() if head_only else m.train()
 head(m,tag).train()
 if tag=='poster':m.face_landback.eval()
