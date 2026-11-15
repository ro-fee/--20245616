"""Prediction requires an already frozen recipe and completed final fitting."""
import argparse
import csv
import gc
import hashlib
import json

from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from common import (ROOT, CLASSES, build_model, dump, foreground_crop, image_transform,
                    loader, probabilities, restore, seed_everything, square_pad)

class TestImages(Dataset):
    def __init__(self, paths, config):
        self.transform = image_transform(config,False)
        self.images = []
        for path in paths:
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
            image.thumbnail((1400,1400),Image.Resampling.LANCZOS)
            result = square_pad(image if config['view']=='raw' else foreground_crop(image))
            result.thumbnail((768,768),Image.Resampling.LANCZOS)
            self.images.append(result)
    def __len__(self):
        return len(self.images)
    def __getitem__(self,i):
        return self.transform(self.images[i]), -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir',type=Path,required=True)
    parser.add_argument('--template',type=Path,required=True)
    args = parser.parse_args()
    # Require inputs under the authorized run boundary; no desktop answer path permitted.
    test_dir,template = args.test_dir.resolve(),args.template.resolve()
    if not test_dir.is_relative_to(ROOT) or not template.is_relative_to(ROOT):
        raise ValueError('Prediction inputs must stay within the run directory')
    ready = json.loads((ROOT/'ready_to_predict.json').read_text())
    recipe_hash = hashlib.sha256((ROOT/'recipe_frozen.json').read_bytes()).hexdigest()
    if ready['recipe_sha256']!=recipe_hash or ready['test_labels_used']:
        raise ValueError('Recipe was changed after fitting')
    with template.open(encoding='utf-8-sig',newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames!=['ID','Category']:
            raise ValueError('Unexpected submission schema')
        # Read only IDs. Template category placeholders never influence inference.
        ids = [row['ID'] for row in reader]
    if len(set(ids))!=len(ids) or not ids:
        raise ValueError('Duplicate or empty submission IDs')
    paths = [(test_dir / name).resolve() for name in ids]
    if any(not p.is_relative_to(test_dir) or not p.is_file() for p in paths):
        raise ValueError('Missing or unsafe test image ID')
    plan = json.loads((ROOT/'plan.json').read_text())
    total = np.zeros((len(ids),5),dtype=np.float64)
    for item in ready['models']:
        config = item['config']
        seed_everything(item['seed'])
        model = build_model(config).cuda()
        restore(model,config,ROOT/item['checkpoint'])
        data = loader(TestImages(paths,config),plan,False,item['seed'])
        p = probabilities(model,data)
        total += item['weight']*p
        print('Predicted',config['name'],item['seed'],len(ids),flush=True)
        del model,data
        gc.collect()
        torch.cuda.empty_cache()
    np.save(ROOT/'test_probabilities.npy',total)
    with (ROOT/'submission.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ID','Category'])
        writer.writerows((name,CLASSES[int(label)]) for name,label in zip(ids,total.argmax(1)))
    dump(ROOT/'prediction_manifest.json',{'n_test_images':len(ids),'recipe_sha256':recipe_hash,
                                         'submission_sha256':hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest(),
                                         'test_labels_used':False,'per_id_overrides':False})
    print('submission.csv ready; no test score computed',flush=True)

if __name__=='__main__':
    main()
