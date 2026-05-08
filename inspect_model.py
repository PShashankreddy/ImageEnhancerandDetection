import app
import torch

sd = app.gan_model.state_dict()
print('Model state_dict entries:', len(sd))

i = 0
for k, v in sd.items():
    try:
        print(k, tuple(v.shape), 'min', float(v.min()), 'max', float(v.max()), 'mean', float(v.mean()))
    except Exception as e:
        print(k, type(v), e)
    i += 1
    if i >= 20:
        break
