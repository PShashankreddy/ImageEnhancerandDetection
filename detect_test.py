import cv2
import numpy as np
from pathlib import Path
import app

out_dir = Path('debug_outputs')

# Load the input we created earlier
img = cv2.imread(str(out_dir / 'input.jpg'))
if img is None:
    print('Input file not found; run test_enhance.py first')
    exit(1)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

enh = app.enhance_image(img_rgb)

dets_orig = app.detect(img_rgb)
dets_enh = app.detect(enh)

print('Detections original:', len(dets_orig))
print('Detections enhanced:', len(dets_enh))

# Save images with boxes for inspection
from base64 import b64decode

# Reuse draw_boxes to get base64 then decode
orig_b64 = app.draw_boxes(img_rgb, dets_orig)
enh_b64 = app.draw_boxes(enh, dets_enh)

for name, b64s in [('orig_boxes.jpg', orig_b64), ('enh_boxes.jpg', enh_b64)]:
    data = b64s.encode()
    import base64
    img_bytes = base64.b64decode(data)
    with open(out_dir / name, 'wb') as f:
        f.write(img_bytes)

print('Saved debug images to', out_dir)
