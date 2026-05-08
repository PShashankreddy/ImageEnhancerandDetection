import torch
from pathlib import Path

ckpt = list(Path('checkpoints').glob('lpgan_ep*.pth'))
if not ckpt:
    print('No checkpoint found')
    exit(1)
path = str(ckpt[-1])
print('Loading', path)
raw = torch.load(path, map_location='cpu')
print('Raw type:', type(raw))

if isinstance(raw, dict):
    print('Top-level keys:', list(raw.keys()))

    # Try common keys
    for candidate in ['G', 'generator', 'model_state_dict', 'state_dict']:
        if candidate in raw:
            state = raw[candidate]
            print(f"Using key '{candidate}' as state dict. Entries: {len(state)}")
            break
    else:
        state = raw
        print('Assuming top-level dict is state dict. Entries:', len(state))

    # Print first 20 param names and shapes
    i = 0
    for k, v in state.items():
        try:
            print(k, tuple(v.shape), 'min', float(v.min()), 'max', float(v.max()), 'mean', float(v.mean()))
        except Exception as e:
            print(k, type(v), e)
        i += 1
        if i >= 20:
            break
else:
    print('Checkpoint is not a dict; type:', type(raw))
