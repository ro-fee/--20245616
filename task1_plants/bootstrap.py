"""All downloads/cache writes stay under this run directory."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for key, folder in {'HF_HOME':'cache/hf', 'HF_HUB_CACHE':'cache/hf/hub',
                    'TORCH_HOME':'cache/torch', 'XDG_CACHE_HOME':'cache',
                    'MPLCONFIGDIR':'cache/matplotlib', 'TMPDIR':'tmp',
                    'CUDA_CACHE_PATH':'cache/cuda', 'TORCH_EXTENSIONS_DIR':'cache/extensions',
                    'TRITON_CACHE_DIR':'cache/triton', 'TORCHINDUCTOR_CACHE_DIR':'cache/inductor'}.items():
    dest = ROOT / folder
    dest.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(dest)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

def main():
    import hashlib
    import json
    import torch
    import torchvision
    from torchvision.models import ConvNeXt_Base_Weights

    torch.hub.set_dir(str(ROOT / 'cache/torch/hub'))
    print('Downloading official generic weights only', flush=True)
    ConvNeXt_Base_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    print('ConvNeXt weights ready', flush=True)
    repo = ROOT/'vendor/dinov2'
    weight_file = ROOT/'cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth'
    if not weight_file.exists():
        torch.hub.download_url_to_file('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
                                      str(weight_file),progress=True)
    model = torch.hub.load(str(repo), 'dinov2_vitl14', source='local',pretrained=False)
    model.load_state_dict(torch.load(weight_file,map_location='cpu',weights_only=True),strict=True)
    print('DINOv2 weights ready', sum(p.numel() for p in model.parameters()), flush=True)
    records = []
    for p in sorted((ROOT / 'cache/torch/hub/checkpoints').glob('*.pth')):
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        records.append({'file': str(p.relative_to(ROOT)), 'sha256': h.hexdigest()})
    (ROOT / 'pretrained_provenance.json').write_text(json.dumps({
        'torch':torch.__version__, 'torchvision':torchvision.__version__,
        'sources':['https://download.pytorch.org/models/', 'https://github.com/facebookresearch/dinov2',
                   'https://dl.fbaipublicfiles.com/dinov2/'],
        'weights':records}, indent=2))

if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description='Prepare official generic pretraining weights.').parse_args()
    main()
