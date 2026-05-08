import cv2
import numpy as np
from pathlib import Path

# Import app which loads models and defines enhance_image
import app

out_dir = Path('debug_outputs')
out_dir.mkdir(exist_ok=True)

# Create a synthetic foggy image (RGB)
h, w = 480, 640
base = np.full((h, w, 3), 90, dtype=np.uint8)
cv2.circle(base, (w//2, h//2), 120, (200,200,200), -1)
# Add fog overlay
fog = np.full_like(base, 255, dtype=np.uint8)
fog = cv2.GaussianBlur(fog, (101,101), 30)
alpha = 0.6
foggy = cv2.addWeighted(base, 1-alpha, fog, alpha, 0)

# Convert BGR->RGB if needed (we created RGB-like)
img_rgb = foggy.copy()

print('Running enhance_image...')
enh = app.enhance_image(img_rgb)

# Save inputs/outputs
cv2.imwrite(str(out_dir / 'input.jpg'), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(out_dir / 'enhanced.jpg'), cv2.cvtColor(enh, cv2.COLOR_RGB2BGR))

# Print some stats
print('Enhanced shape:', enh.shape)
print('Enhanced min/max:', int(enh.min()), int(enh.max()))

print('Done. Files written to', out_dir)
