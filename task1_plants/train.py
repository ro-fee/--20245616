import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from common import (ROOT, EMA, TrainImages, build_model, checkpoint, dump, loader,
                    metrics, optimizer_for, probabilities, rank, restore, seed_everything)

def fit(config, plan, records, indexes, output, seed, validation=None, fixed_epochs=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output/'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['config']!=config or result['seed']!=seed or result['train_indexes']!=indexes:
            raise ValueError('Existing run has different inputs')
        return result
    seed_everything(seed)
    model = build_model(config).cuda()
    optimizer = optimizer_for(model, config)
    ema = EMA(model)
    training = loader(TrainImages(records,indexes,config,True),plan,True,seed)
    val_loader = None
    if validation is not None:
        val_loader = loader(TrainImages(records,validation,config,False),plan,False,seed)
        y = [records[i]['label'] for i in validation]
    criterion = nn.CrossEntropyLoss(label_smoothing=.03)
    epochs = fixed_epochs if fixed_epochs is not None else config['epochs']
    # Refits retain the original planned cosine schedule and stop at the selected epoch.
    schedule_epochs = config['epochs']
    best, best_epoch = None, 0
    logs = []
    trainable = [p for p in model.parameters() if p.requires_grad]
    (output/'epochs.jsonl').write_text('',encoding='utf-8')
    print(json.dumps({'event':'run_start','experiment':config['name'],'seed':seed,
                      'n_train':len(indexes),'n_validation':len(validation or []),'epochs':epochs,
                      'trainable_parameters':sum(p.numel() for p in trainable)}),flush=True)
    for epoch in range(1,epochs+1):
        started = time.monotonic()
        model.train()
        total_loss, total_correct, count = 0.0,0,0
        for step,(images,labels) in enumerate(training):
            progress = epoch-1 + step/len(training)
            factor = min(1.0,(progress+.2)/3)
            if progress>=3:
                factor = .1 + .9*(1+math.cos(math.pi*(progress-3)/max(1,schedule_epochs-3)))/2
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr']*factor
            images, labels = images.cuda(non_blocking=True),labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits.float(),labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Training loss became nonfinite')
            loss.backward()
            nn.utils.clip_grad_norm_(trainable,1.0)
            optimizer.step()
            ema.update()
            total_loss += float(loss.detach())*len(labels)
            total_correct += int((logits.argmax(1)==labels).sum())
            count += len(labels)
        log = {'epoch':epoch,'train_loss':total_loss/count,'train_accuracy':total_correct/count,
               'seconds':time.monotonic()-started}
        if val_loader is not None:
            with ema.apply():
                p = probabilities(model,val_loader)
                m = metrics(y,p)
                if best is None or rank(m)>rank(best):
                    best,best_epoch = m,epoch
                    checkpoint(model,config,epoch,output/'model.pt')
                    np.save(output/'validation_probabilities.npy',p)
            log.update({'val_accuracy':m['accuracy'],'val_macro_f1':m['macro_f1'],
                        'val_nll':m['nll'],'best_epoch':best_epoch})
        logs.append(log)
        print(json.dumps({'event':'epoch','experiment':config['name'],'run':output.name,**log}),flush=True)
        with (output/'epochs.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(log)+'\n')
        dump(ROOT/'status.json',{'stage':'training','experiment':config['name'],'run':str(output.relative_to(ROOT)),**log})
        if validation is not None and epoch>=config['min_epochs'] and epoch-best_epoch>=config['patience']:
            break
    if validation is None:
        with ema.apply():
            checkpoint(model,config,epochs,output/'model.pt')
        best_epoch = epochs
    result = {'config':config,'seed':seed,'train_indexes':indexes,'validation_indexes':validation,
              'best_epoch':best_epoch,'metrics':best,'epochs_ran':len(logs)}
    dump(result_path,result)
    del model,ema,optimizer,training,val_loader,trainable
    gc.collect()
    torch.cuda.empty_cache()
    return result

def evaluate_checkpoint(config, plan, records, indexes, filename):
    model = build_model(config).cuda()
    restore(model,config,filename)
    data_loader = loader(TrainImages(records,indexes,config,False),plan,False,plan['seed'])
    result = probabilities(model,data_loader)
    del model,data_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result
