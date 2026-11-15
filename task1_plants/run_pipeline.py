"""Development tuning -> recipe lock -> one audit -> full training. No test access."""
import hashlib
import json
import traceback

import numpy as np
from common import ROOT, dump, make_splits, metrics, prepare_training_cache, rank
from train import fit, evaluate_checkpoint


def main():
    plan = json.loads((ROOT/'plan.json').read_text())
    sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}
    sources['plan.json'] = hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest()
    source_path = ROOT/'source_manifest.json'
    if source_path.exists() and json.loads(source_path.read_text())!=sources:
        raise ValueError('Run sources changed after launch; create a new clean run instead of silently reusing results')
    dump(source_path,sources)
    records,splits = make_splits(plan)
    prepare_training_cache(records)
    f0 = splits['folds'][0]
    screening = {}
    for config in plan['experiments']:
        screening[config['name']] = fit(config,plan,records,f0['train'],ROOT/'runs'/config['name']/'fold0',
                                        plan['seed'],validation=f0['validation'])
    cnn = max((r for r in screening.values() if r['config']['architecture']!='dinov2_large'),
               key=lambda r:rank(r['metrics']))['config']
    dino = next(r['config'] for r in screening.values() if r['config']['architecture']=='dinov2_large')
    cv = {}
    dev_indexes = splits['development']
    dev_positions = {i:j for j,i in enumerate(dev_indexes)}
    y_dev = [records[i]['label'] for i in dev_indexes]
    for config in (cnn,dino):
        oof = np.full((len(dev_indexes),5),np.nan,dtype=np.float32)
        fold_results = []
        for fold,part in enumerate(splits['folds']):
            directory = ROOT/'runs'/config['name']/f'fold{fold}'
            result = fit(config,plan,records,part['train'],directory,plan['seed']+fold,
                         validation=part['validation'])
            p = np.load(directory/'validation_probabilities.npy')
            oof[[dev_positions[i] for i in part['validation']]] = p
            fold_results.append(result)
        if not np.isfinite(oof).all():
            raise ValueError('OOF coverage is incomplete')
        np.save(ROOT/f'oof_{config["name"]}.npy',oof)
        cv[config['name']] = {'config':config,'metrics':metrics(y_dev,oof),'folds':fold_results,
                             'refit_epochs':int(round(float(np.median([r['best_epoch'] for r in fold_results]))))}
    cnn_oof = np.load(ROOT/f'oof_{cnn["name"]}.npy')
    dino_oof = np.load(ROOT/f'oof_{dino["name"]}.npy')
    weights = []
    for w in plan['ensemble_weights']:
        weights.append({'dino_weight':w,'metrics':metrics(y_dev,w*dino_oof+(1-w)*cnn_oof)})
    chosen = max(weights,key=lambda r:rank(r['metrics']))
    selected_models = []
    for config,w in ((cnn,1-chosen['dino_weight']),(dino,chosen['dino_weight'])):
        if w>0:
            selected_models.append({'config':config,'weight':w,'epochs':cv[config['name']]['refit_epochs']})
    selection = {'dino_weight':chosen['dino_weight'],'models':selected_models,
                 'development_metrics':chosen['metrics'],'weight_candidates':weights,
                 'final_seeds':plan['final_seeds'],'selection_data':'400-image development set only',
                 'audit_results_seen_at_selection':False,'test_images_seen_at_selection':False,
                 'source_manifest_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest()}
    frozen = ROOT/'recipe_frozen.json'
    if frozen.exists() and json.loads(frozen.read_text())!=selection:
        raise ValueError('Frozen recipe cannot be overwritten with a different selection')
    dump(frozen,selection)
    dump(ROOT/'development_summary.json',{'screening':screening,'cv':cv,'selection':selection})
    print(json.dumps({'event':'recipe_frozen','dino_weight':chosen['dino_weight'],
                      'development_accuracy':chosen['metrics']['accuracy'],
                      'development_macro_f1':chosen['metrics']['macro_f1']}),flush=True)
    audit_path = ROOT/'audit_summary.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
    else:
        audit_probabilities = np.zeros((len(splits['audit']),5),dtype=np.float64)
        individual = {}
        for item in selected_models:
            config = item['config']
            directory = ROOT/'runs'/config['name']/'audit_refit'
            assert not set(dev_indexes)&set(splits['audit'])
            fit(config,plan,records,dev_indexes,directory,plan['seed']+100,fixed_epochs=item['epochs'])
            p = evaluate_checkpoint(config,plan,records,splits['audit'],directory/'model.pt')
            np.save(directory/'audit_probabilities.npy',p)
            # Evaluation only; no resulting value controls subsequent choices.
            y_audit = [records[i]['label'] for i in splits['audit']]
            individual[config['name']] = metrics(y_audit,p)
            audit_probabilities += item['weight']*p
        audit = {'individual':individual,'ensemble':metrics(y_audit,audit_probabilities),
                 'recipe_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
                 'used_for_selection':False}
        np.save(ROOT/'audit_ensemble_probabilities.npy',audit_probabilities)
        dump(audit_path,audit)

    print(json.dumps({'event':'independent_audit','metrics':audit['ensemble']}),flush=True)
    final_models = []
    for item in selected_models:
        for seed in plan['final_seeds']:
            directory = ROOT/'runs'/item['config']['name']/f'final_seed{seed}'
            fit(item['config'],plan,records,list(range(500)),directory,seed,fixed_epochs=item['epochs'])
            final_models.append({'config':item['config'],'weight':item['weight']/len(plan['final_seeds']),
                                 'checkpoint':str((directory/'model.pt').relative_to(ROOT)),
                                 'seed':seed,'epochs':item['epochs']})
    ready = {'models':final_models,'recipe_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
             'training_samples':500,'test_labels_used':False}
    dump(ROOT/'ready_to_predict.json',ready)
    dump(ROOT/'status.json',{'stage':'ready_to_predict','audit_accuracy':audit['ensemble']['accuracy'],
                            'audit_macro_f1':audit['ensemble']['macro_f1']})
    print('READY_TO_PREDICT',flush=True)

if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        main()
    except Exception:
        dump(ROOT/'status.json',{'stage':'failed','traceback':traceback.format_exc()})
        raise
