"""Specialist transfer learning. Reads canonical training images only."""
import argparse,math,signal,time,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as T
from PIL import Image
from common import EXP,CFG,GIB,data,disk,emit,identity,metric,probability,seed,sha,write
from models import model,head,mode,official_load,base_model

STOP=False
def request_stop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
class Faces(Dataset):
 def __init__(self,pixels,labels,idx,size,augment=False):
  self.pixels=pixels;self.labels=labels;self.idx=idx
  ops=[T.Resize((size,size),interpolation=T.InterpolationMode.BILINEAR)]
  if augment:ops += [T.RandomHorizontalFlip(),T.RandomAffine(6,translate=(.03,.03),scale=(.95,1.05)),T.ColorJitter(.1,.1)]
  ops += [T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])]
  self.tf=T.Compose(ops)
 def __len__(self):return len(self.idx)
 def __getitem__(self,i):
  j=self.idx[i];return self.tf(Image.fromarray(self.pixels[j]).convert('RGB')),int(self.labels[j])
def loader(ds,batch,shuffle=False,generator=None):
 return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=2,pin_memory=True,
                   generator=generator,persistent_workers=False,drop_last=False)
def setup(tag):
 if not torch.cuda.is_available():raise RuntimeError('Training requires a CUDA GPU')
 cfg=CFG['models'][tag];free,total=torch.cuda.mem_get_info()
 if free/GIB<cfg['gpu_required_gib']:raise RuntimeError('GPU_RESERVE: insufficient free memory before model startup')
 torch.cuda.set_per_process_memory_fraction(cfg['gpu_cap_gib']*GIB/total,0)
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
 assert torch.cuda.is_bf16_supported();disk(2)
 emit('gpu_ready',model=tag,free_gib=free/GIB,cap_gib=cfg['gpu_cap_gib'])
def cpu_state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def save(path,obj):
 disk(2);tmp=Path(path).with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)

def finish(run,ident,state,metrics,epoch,best_epoch,prediction_path,elapsed):
 """Commit the final view from a genuine completed epoch, including resume at the last boundary."""
 arr=np.load(prediction_path);df,_,va,_=data(ident['fold'])
 assert np.array_equal(arr['valid_idx'],va)
 for name in ('plain','flip'):
  probability(arr[name],len(va));assert metric(arr[name],df.label.to_numpy()[va])==metrics[name]
 final=run/'final.npz';tmp=run/'final.npz.tmp';shutil.copyfile(prediction_path,tmp);tmp.replace(final)
 save(run/'final.pt',dict(state=state,identity=ident,epoch=epoch))
 write(run/'complete.json',dict(identity=ident,best_epoch=best_epoch,epoch=epoch,metrics=metrics,
  elapsed_seconds=elapsed,checkpoint_sha256=sha(run/'final.pt'),prediction_sha256=sha(final),test_read=False,answer_read=False))
def trainable(m,tag,head_only):
 for p in m.parameters():p.requires_grad=not head_only
 for p in head(m,tag).parameters():p.requires_grad=True
 if tag=='poster':
  for p in m.face_landback.parameters():p.requires_grad=False
 mode(m,tag,head_only)
def optimizer(m,tag):
 cfg=CFG['models'][tag];ids={id(p) for p in head(m,tag).parameters()}
 return torch.optim.AdamW([
  {'params':[p for p in m.parameters() if id(p) not in ids], 'lr':cfg['backbone_lr'],'base_lr':cfg['backbone_lr']},
  {'params':list(head(m,tag).parameters()),'lr':cfg['head_lr'],'base_lr':cfg['head_lr']}],weight_decay=CFG['weight_decay'])
@torch.no_grad()
def predict(m,dl):
 m.eval();plain=[];flip=[]
 for x,_ in dl:
  if STOP:raise InterruptedError('User stop requested during validation')
  x=x.cuda(non_blocking=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):a=m(x);b=m(x.flip(-1))
  pa=a.float().softmax(1);pb=b.float().softmax(1)
  plain.append(pa.cpu().numpy());flip.append(((pa+pb)/2).cpu().numpy())
 return {'plain':np.concatenate(plain),'flip':np.concatenate(flip)}
def smoke(tag):
 ident=identity(tag,0,CFG['epochs']);run=EXP/f'artifacts/smoke_{tag}';run.mkdir(exist_ok=True)
 if (run/'complete.json').exists():
  assert __import__('json').loads((run/'complete.json').read_text())['identity']==ident;emit('smoke_already_complete',model=tag);return
 setup(tag);seed(CFG['seed']);df,tr,va,pixels=data(0);cfg=CFG['models'][tag]
 dl=loader(Faces(pixels,df.label.to_numpy(),tr[:cfg['batch_size']],cfg['image_size']),cfg['batch_size'])
 m=model(tag).cuda();trainable(m,tag,False);opt=optimizer(m,tag)
 frozen={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad}
 frozen_buffers={n:v.detach().cpu().clone() for n,v in m.named_buffers() if tag=='poster' and n.startswith('face_landback.')}
 x,y=next(iter(dl));x=x.cuda();y=y.cuda();started=time.monotonic()
 original_equivalence=None
 if tag=='hsemotion':
  raw,_=official_load(EXP/'vendor/hsemotion/models/affectnet_emotions/enet_b2_7.pt',tag)
  original=raw.cuda().eval();rebuilt=base_model(tag).cuda().eval()
  rebuilt.load_state_dict(torch.load(EXP/'weights/hsemotion_state.pt',map_location='cpu',weights_only=True),strict=True)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):oa=original(x[:2]).float();ob=rebuilt(x[:2]).float()
  assert torch.allclose(oa,ob,atol=1e-4,rtol=1e-4),'Official model forward differs from reconstruction'
  original_equivalence=True;del original,raw,rebuilt;torch.cuda.empty_cache()
 for step in range(10):
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=nn.functional.cross_entropy(m(x),y)
  assert torch.isfinite(loss);loss.backward()
  grads=[p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
  assert grads and all(torch.isfinite(g).all() for g in grads) and any(torch.count_nonzero(g)>0 for g in grads)
  torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 assert all(torch.equal(v,dict(m.named_parameters())[n].detach().cpu()) for n,v in frozen.items())
 assert all(torch.equal(v,dict(m.named_buffers())[n].detach().cpu()) for n,v in frozen_buffers.items())
 m.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):a=m(x[:2]).float();b=m(x[:2].flip(-1)).float()
 save(run/'reload.pt',{'state':cpu_state(m),'identity':ident})
 m.load_state_dict(torch.load(run/'reload.pt',map_location='cpu',weights_only=True)['state'],strict=True)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):c=m(x[:2]).float()
 assert torch.allclose(a,c,atol=1e-4,rtol=1e-4) and not torch.allclose(a,b,atol=1e-4,rtol=1e-4)
 peak=torch.cuda.max_memory_reserved()/GIB
 write(run/'complete.json',dict(identity=ident,steps=10,finite_gradients=True,frozen_parameters_unchanged=True,
      official_reconstruction_equal=original_equivalence,frozen_buffers_unchanged=True,reload_equal=True,image_affects_output=True,peak_reserved_gib=peak,elapsed_seconds=time.monotonic()-started))
 emit('smoke_complete',model=tag,peak_gib=peak)
def train(tag,fold,stop_epoch=None):
 stop_epoch=stop_epoch or CFG['epochs'];ident=identity(tag,fold,stop_epoch)
 run=EXP/f'artifacts/{tag}_fold{fold}';run.mkdir(exist_ok=True)
 if (run/'complete.json').exists():
  assert __import__('json').loads((run/'complete.json').read_text())['identity']==ident;emit('already_complete',model=tag,fold=fold);return
 setup(tag);df,tr,va,pixels=data(fold);y=df.label.to_numpy();cfg=CFG['models'][tag]
 ds=Faces(pixels,y,tr,cfg['image_size'],True);valid=loader(Faces(pixels,y,va,cfg['image_size']),cfg['batch_size'])
 seed(CFG['seed']+fold);m=model(tag).cuda();opt=optimizer(m,tag);start_epoch=0;best=-1.;best_epoch=0
 if (run/'last.pt').exists():
  old=torch.load(run/'last.pt',map_location='cpu',weights_only=True);assert old['identity']==ident
  m.load_state_dict(old['state'],strict=True);opt.load_state_dict(old['optimizer']);start_epoch=old['epoch'];best=old['best'];best_epoch=old['best_epoch']
  assert 0<start_epoch<=stop_epoch
  pred_path=run/old['prediction_file'];assert sha(pred_path)==old['prediction_sha256']
  metrics=old['metrics'];state=old['state']
  emit('resuming',model=tag,fold=fold,completed_epoch=start_epoch)
 started=time.monotonic()
 for epoch in range(start_epoch,stop_epoch):
  seed(CFG['seed']+fold*100+epoch);trainable(m,tag,epoch<CFG['head_warmup_epochs'])
  dl=loader(ds,cfg['batch_size'],True,torch.Generator().manual_seed(CFG['seed']+fold*100+epoch))
  accum=cfg['accumulation'];opt.zero_grad(set_to_none=True);loss_total=0.;n=0
  for i,(x,target) in enumerate(dl):
   if CFG.get('deadline_utc') and datetime.now(timezone.utc)>=datetime.fromisoformat(CFG['deadline_utc']):raise InterruptedError('Experiment freeze deadline reached; latest full epoch retained')
   if STOP or (EXP/'STOP_REQUESTED').exists():raise InterruptedError('User stop requested; latest full epoch retained')
   frac=(epoch+i/len(dl))/CFG['epochs'];mult=.1+.9*.5*(1+math.cos(math.pi*frac))
   for group in opt.param_groups:group['lr']=group['base_lr']*mult
   x=x.cuda(non_blocking=True);target=target.cuda(non_blocking=True)
   window=min(accum,len(dl)-(i//accum)*accum)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=nn.functional.cross_entropy(m(x),target,label_smoothing=CFG['label_smoothing'])
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   (loss/window).backward();loss_total+=float(loss.detach())*len(x);n+=len(x)
   if (i+1)%accum==0 or i+1==len(dl):
    norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1)
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient norm')
    opt.step();opt.zero_grad(set_to_none=True)
   if i==0 or (i+1)%100==0:emit('train_progress',model=tag,fold=fold,epoch=epoch+1,batch=i+1,total=len(dl),loss=loss_total/n,peak_gib=torch.cuda.max_memory_reserved()/GIB,elapsed_seconds=time.monotonic()-started)
  probs=predict(m,valid)
  for p in probs.values():probability(p,len(va))
  metrics={mode:metric(p,y[va]) for mode,p in probs.items()};score=metrics['plain']['accuracy']
  state=cpu_state(m)
  pred_path=run/f'epoch{epoch+1}.npz';pred_tmp=run/f'epoch{epoch+1}.npz.tmp'
  with pred_tmp.open('wb') as f:np.savez_compressed(f,valid_idx=va,**probs)
  pred_tmp.replace(pred_path)
  if score>best:
   best=score;best_epoch=epoch+1
   save(run/'best.pt',dict(state=state,identity=ident,epoch=epoch+1))
   shutil.copyfile(pred_path,run/'best.npz')
   write(run/'best.json',dict(identity=ident,epoch=epoch+1,selection='single model plain validation accuracy',metrics=metrics,
     checkpoint_sha256=sha(run/'best.pt'),prediction_sha256=sha(run/'best.npz')))
  write(run/f'epoch{epoch+1}.json',dict(identity=ident,epoch=epoch+1,metrics=metrics,training_loss=loss_total/n))
  save(run/'last.pt',dict(state=state,optimizer=opt.state_dict(),identity=ident,epoch=epoch+1,best=best,best_epoch=best_epoch,
   metrics=metrics,prediction_file=pred_path.name,prediction_sha256=sha(pred_path)))
  emit('epoch_complete',model=tag,fold=fold,epoch=epoch+1,metrics=metrics)
 finish(run,ident,state,metrics,stop_epoch,best_epoch,pred_path,time.monotonic()-started)
 emit('train_complete',model=tag,fold=fold)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','train']);p.add_argument('--model',choices=list(CFG['models']),required=True)
 p.add_argument('--fold',type=int,default=0);p.add_argument('--stop-epoch',type=int);a=p.parse_args()
 try:smoke(a.model) if a.mode=='smoke' else train(a.model,a.fold,a.stop_epoch)
 except InterruptedError as e:
  write(EXP/'artifacts/interrupted.json',dict(reason=str(e),model=a.model,fold=a.fold,test_read=False));raise SystemExit(143)
