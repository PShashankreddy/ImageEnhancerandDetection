import cv2
import numpy as np
from pathlib import Path
import app

out_dir = Path('debug_outputs')
out_dir.mkdir(exist_ok=True)

# synthetic foggy image
h, w = 480, 640
base = np.full((h, w, 3), 85, dtype=np.uint8)
cv2.rectangle(base, (140, 140), (500, 340), (170, 170, 170), -1)
fog = np.full_like(base, 255, dtype=np.uint8)
fog = cv2.GaussianBlur(fog, (91, 91), 30)
foggy = cv2.addWeighted(base, 0.35, fog, 0.65, 0)

img_rgb = foggy

enh_nat = app.enhance_image(img_rgb, strength=0.15)
enh_str = app.enhance_image(img_rgb, strength=0.9)

cv2.imwrite(str(out_dir / 'strength_input.jpg'), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(out_dir / 'strength_natural.jpg'), cv2.cvtColor(enh_nat, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(out_dir / 'strength_strong.jpg'), cv2.cvtColor(enh_str, cv2.COLOR_RGB2BGR))

print('Natural min/max/std:', int(enh_nat.min()), int(enh_nat.max()), float(enh_nat.std()))
print('Strong  min/max/std:', int(enh_str.min()), int(enh_str.max()), float(enh_str.std()))
print('Saved strength comparison images to', out_dir)
