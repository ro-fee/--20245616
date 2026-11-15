"""Fixed 30-checkpoint FER ensemble for unlabeled images."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

TAGS=['efficientnet_b0','convnext_tiny','convnext_base','dinov2_large','hsemotion','poster']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path,obj):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(p)

def verify_assets(assets):
    m=read(assets/'assets_manifest.json')
    expected={(tag,fold) for tag in TAGS for fold in range(5)}
    if {(r['model'],r['fold']) for r in m['checkpoints']}!=expected or len(m['checkpoints'])!=30:
        raise ValueError('Exactly 30 frozen checkpoints are required')
    for r in m['checkpoints']+m['files']:
        p=(assets/r['path']).resolve()
        if not p.is_relative_to(assets.resolve()) or not p.is_file() or sha(p)!=r['sha256']:
            raise ValueError('Missing or modified asset: '+r['path'])
    print('ASSETS_VERIFIED: 30 checkpoints and frozen supporting files',flush=True)
    return m

def prepare_pixels(test_dir,template_path,out):
    import numpy as np
    from PIL import Image
    with Path(template_path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['ID','Emotion']:
            raise ValueError('Expected template columns ID,Emotion')
        ids=[r['ID'] for r in reader]
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('Template IDs must be unique and nonempty')
    root=Path(test_dir).resolve();pixels=[]
    for image_id in ids:
        if '/' in image_id or '\\' in image_id or Path(image_id).name!=image_id:
            raise ValueError('Expected an image filename, not a path')
        path=(root/image_id).resolve()
        if not path.is_relative_to(root):raise ValueError('Image escaped test directory')
        with Image.open(path) as im:
            a=np.asarray(im.convert('L'))
        if a.shape!=(48,48):raise ValueError('Expected 48x48 image: '+image_id)
        pixels.append(a)
    np.save(out/'test_pixels.npy',np.stack(pixels))
    save(out/'image_ids.json',ids)
    return ids


def build_model(tag,fold,assets,dtype):
    import torch
    from torch import nn
    from torchvision import models
    checkpoint=assets/'weights'/f'{tag}_fold{fold}.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if tag in TAGS[:3]:
        assert state['fold']==fold and state['model']==tag
        m=getattr(models,tag)(weights=None)
        idx=1 if tag=='efficientnet_b0' else 2
        m.classifier[idx]=nn.Linear(m.classifier[idx].in_features,6)
        m.load_state_dict(state['state_dict'],strict=True)
        return m.to(memory_format=torch.channels_last)
    if tag=='hsemotion':
        import timm
        m=timm.create_model('tf_efficientnet_b2',pretrained=False,num_classes=6,global_pool='max')
    elif tag=='poster':
        sys.path.insert(0,str(assets))
        from poster_models.PosterV2_7cls import pyramid_trans_expr2
        m=pyramid_trans_expr2(img_size=224,num_classes=7)
        replaced=False
        for name,layer in list(m.VIT.head.named_modules()):
            if isinstance(layer,nn.Linear) and layer.out_features==7:
                parts=name.split('.');parent=m.VIT.head
                for part in parts[:-1]:parent=getattr(parent,part)
                setattr(parent,parts[-1],nn.Linear(layer.in_features,6));replaced=True;break
        assert replaced
    else:
        from transformers import AutoModel
        from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
        ident=state['identity'];assert ident['fold']==fold
        assert ident['revision']=='47b73eefe95e8d44ec3623f8890bd894b6ea2d6c'
        class DinoClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                base=AutoModel.from_pretrained(str(assets/'dinov2_base'),local_files_only=True,
                    torch_dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True)
                self.registers=getattr(base.config,'num_register_tokens',0)
                dim=base.config.hidden_size*2
                self.backbone=get_peft_model(base,LoraConfig(r=ident['rank'],lora_alpha=ident['alpha'],
                    lora_dropout=ident['lora_dropout'],target_modules=['query','value'],bias='none'))
                self.head=nn.Sequential(nn.LayerNorm(dim),nn.Dropout(.1),nn.Linear(dim,6))
            def forward(self,x):
                t=self.backbone(pixel_values=x).last_hidden_state
                return self.head(torch.cat((t[:,0],t[:,1+self.registers:].mean(1)),dim=-1).float())
        m=DinoClassifier()
        current=get_peft_model_state_dict(m.backbone)
        assert set(current)==set(state['adapter'])
        for k in current:assert current[k].shape==state['adapter'][k].shape
        result=set_peft_model_state_dict(m.backbone,state['adapter'])
        assert not result.unexpected_keys
        m.head.load_state_dict(state['head'],strict=True)
        return m
    assert state['identity']['model']==tag and state['identity']['fold']==fold
    assert state['epoch']==(10 if tag=='hsemotion' else 24)
    m.load_state_dict(state['state'],strict=True)
    return m

def worker(args):
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader,Dataset
    from torchvision import transforms as T
    tag,fold=args.worker;fold=int(fold);out=args.output
    manifest=read(args.assets/'assets_manifest.json')
    rec=next(r for r in manifest['checkpoints'] if r['model']==tag and r['fold']==fold)
    assert sha(args.assets/rec['path'])==rec['sha256']
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float32
    torch.set_num_threads(2);torch.manual_seed(20261011)
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
    size={'efficientnet_b0':160,'convnext_tiny':160,'convnext_base':160,'dinov2_large':224,'hsemotion':260,'poster':224}[tag]
    interp=T.InterpolationMode.BILINEAR if tag in ['hsemotion','poster'] else T.InterpolationMode.BICUBIC
    tf=T.Compose([T.Resize((size,size),interpolation=interp),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    pixels=np.load(out/'test_pixels.npy',mmap_mode='r')
    class Faces(Dataset):
        def __len__(self):return len(pixels)
        def __getitem__(self,i):return tf(Image.fromarray(pixels[i]).convert('RGB'))
    batch=8 if tag in ['dinov2_large','poster'] else 16
    dl=DataLoader(Faces(),batch_size=batch,shuffle=False,num_workers=0)
    m=build_model(tag,fold,args.assets,dtype).cuda().eval();pred=[];started=time.monotonic()
    flip=tag in ['efficientnet_b0','convnext_tiny','poster']
    with torch.inference_mode():
        for n,x in enumerate(dl):
            x=x.cuda()
            if tag in TAGS[:3]:x=x.contiguous(memory_format=torch.channels_last)
            if tag=='dinov2_large':x=x.to(dtype)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
                a=m(x);b=m(x.flip(-1)) if flip else None
            p=a.float().softmax(1)
            if b is not None:p=(p+b.float().softmax(1))/2
            pred.append(p.cpu().numpy())
            if (n+1)%100==0:print(tag,fold,f'{n+1}/{len(dl)} batches',flush=True)
    probs=np.concatenate(pred)
    assert probs.shape==(len(read(out/'image_ids.json')),6) and np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-5)
    path=out/f'{tag}_fold{fold}.npy';np.save(path,probs)
    save(out/f'{tag}_fold{fold}.json',dict(checkpoint_sha256=rec['sha256'],prediction_sha256=sha(path),
        run_identity=read(out/'run_identity.json'),seconds=time.monotonic()-started,dtype=str(dtype)))
    print('FOLD_COMPLETE',tag,fold,flush=True)

def main(args):
    import numpy as np
    import torch
    import importlib.metadata as metadata
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this ensemble')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    manifest=verify_assets(args.assets)
    ids=prepare_pixels(args.test_dir,args.template,out)
    identity=dict(assets_manifest_sha256=sha(args.assets/'assets_manifest.json'),code_sha256=sha(__file__),
        pixels_sha256=sha(out/'test_pixels.npy'),gpu=torch.cuda.get_device_name(),
        precision='bfloat16' if torch.cuda.is_bf16_supported(including_emulation=False) else 'float32',
        versions={p:metadata.version(p) for p in ['torch','torchvision','transformers','peft','timm','numpy','Pillow']})
    ip=out/'run_identity.json'
    if ip.exists() and read(ip)!=identity:raise ValueError('Run identity changed; use a new output directory')
    save(ip,identity);print(json.dumps(identity,indent=2),flush=True)
    means={}
    for tag in TAGS:
        parts=[]
        for fold in range(5):
            path=out/f'{tag}_fold{fold}.npy';receipt=out/f'{tag}_fold{fold}.json'
            rec=next(r for r in manifest['checkpoints'] if r['model']==tag and r['fold']==fold)
            cached=path.exists() and receipt.exists()
            if cached:
                r=read(receipt)
                cached=r['run_identity']==identity and r['checkpoint_sha256']==rec['sha256'] and r['prediction_sha256']==sha(path)
            if not cached:
                subprocess.run([sys.executable,'-B',__file__,'--assets',str(args.assets),'--test-dir',str(args.test_dir),'--template',str(args.template),
                    '--output',str(out),'--worker',tag,str(fold)],check=True)
            p=np.load(path);assert p.shape==(len(read(out/'image_ids.json')),6) and np.isfinite(p).all();parts.append(p)
        means[tag]=np.mean(parts,axis=0);np.save(out/f'{tag}_mean.npy',means[tag])
    cnn=.7*(.4*means['efficientnet_b0']+.6*means['convnext_tiny'])+.3*means['convnext_base']
    dino=.5*means['dinov2_large']+.5*cnn
    final=.5*dino+.25*means['hsemotion']+.25*means['poster']
    assert np.allclose(final.sum(1),1,atol=1e-5)
    np.save(out/'final_probs.npy',final)
    with (out/'submission.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['ID','Emotion']);w.writerows(zip(ids,final.argmax(1).tolist()))
    save(out/'completion.json',dict(status='complete',rows=len(ids),classes=manifest['classes'],
        weights=manifest['weights'],run_identity=identity,csv_sha256=sha(out/'submission.csv'),
        prediction_rows=len(ids)))
    print('COMPLETE:',out/'submission.csv',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--test-dir',type=Path,required=True);p.add_argument('--template',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',nargs=2);args=p.parse_args()
    worker(args) if args.worker else main(args)
