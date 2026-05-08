# app.py - Complete Fixed Flask Web App

from flask import Flask, render_template, request, jsonify
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import base64
from ultralytics import YOLO

def calc_psnr(img1, img2, data_range=255):
    # OpenCV provides PSNR; fall back to manual formula
    try:
        return float(cv2.PSNR(img1, img2))
    except Exception:
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(data_range / np.sqrt(mse))

def calc_ssim(img1, img2, data_range=255):
    # Simple SSIM (grayscale) implementation compatible with skimage interface
    try:
        if img1.ndim == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2

        img1_gray = img1_gray.astype(np.float64)
        img2_gray = img2_gray.astype(np.float64)

        K1 = 0.01
        K2 = 0.03
        L = data_range
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        # Gaussian blur for means
        mu1 = cv2.GaussianBlur(img1_gray, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2_gray, (11, 11), 1.5)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1_gray * img1_gray, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2_gray * img2_gray, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1_gray * img2_gray, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(np.mean(ssim_map))
    except Exception:
        return 0.0
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
device = 'cpu'

# ═══════════════════════════════════════════════════════════════════════════
# GAN CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class DWSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            DWSConvBlock(ch, ch), 
            nn.Conv2d(ch, ch, 1, bias=False), 
            nn.InstanceNorm2d(ch, affine=True)
        )
    def forward(self, x): return x + self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            DWSConvBlock(in_ch, out_ch)
        )
    def forward(self, x): return self.block(x)

class LPGANGenerator(nn.Module):
    def __init__(self, base_ch=32, n_res=6):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_ch, 7, 1, 3, bias=False), 
            nn.InstanceNorm2d(base_ch, affine=True), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = DWSConvBlock(base_ch, base_ch*2, stride=2)
        self.enc3 = DWSConvBlock(base_ch*2, base_ch*4, stride=2)
        self.enc4 = DWSConvBlock(base_ch*4, base_ch*8, stride=2)
        self.bottleneck = nn.Sequential(*[ResBlock(base_ch*8) for _ in range(n_res)])
        self.dec4 = UpBlock(base_ch*8 + base_ch*8, base_ch*4)
        self.dec3 = UpBlock(base_ch*4 + base_ch*4, base_ch*2)
        self.dec2 = UpBlock(base_ch*2 + base_ch*2, base_ch)
        self.out = nn.Sequential(
            nn.Conv2d(base_ch + base_ch, base_ch, 3, 1, 1, bias=False), 
            nn.InstanceNorm2d(base_ch, affine=True), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(base_ch, 3, 7, 1, 3), 
            nn.Tanh()
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.dec4(torch.cat([b, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        return self.out(torch.cat([d2, e1], dim=1))

# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════

print("Loading models...")
yolo_model = YOLO('yolov8s.pt')
gan_model = LPGANGenerator(base_ch=32, n_res=6).to(device)

# Load checkpoint (robust loader: supports multiple checkpoint shapes and DataParallel 'module.' prefixes)
ckpt_dir = Path('./checkpoints')
ckpt_loaded = False
if ckpt_dir.exists():
    ckpt_files = sorted(ckpt_dir.glob('lpgan_ep*.pth'))
    if ckpt_files:
        ckpt_path = str(ckpt_files[-1])
        try:
            raw = torch.load(ckpt_path, map_location=device)

            # Resolve common checkpoint containers
            if isinstance(raw, dict):
                if 'G' in raw:
                    state = raw['G']
                elif 'generator' in raw:
                    state = raw['generator']
                elif 'model_state_dict' in raw:
                    state = raw['model_state_dict']
                elif 'state_dict' in raw:
                    state = raw['state_dict']
                else:
                    # assume this dict is already the state_dict
                    state = raw
            else:
                state = raw

            # Remove 'module.' prefix if present (from DataParallel)
            try:
                from collections import OrderedDict
                new_state = OrderedDict()
                for k, v in state.items():
                    new_key = k.replace('module.', '') if k.startswith('module.') else k
                    new_state[new_key] = v
                gan_model.load_state_dict(new_state)
            except Exception:
                # Fallback: try to load state directly
                gan_model.load_state_dict(state)

            ckpt_loaded = True
            print(f"✅ Loaded GAN checkpoint: {Path(ckpt_path).name}")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint ({Path(ckpt_path).name}): {e}")

gan_model.eval()
print("✅ All models loaded!")

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def classical_defog_enhance(img_rgb, strength=0.5):
    """Natural-to-strong dehazing fallback controlled by strength in [0,1]."""
    strength = float(np.clip(strength, 0.0, 1.0))
    img = img_rgb.astype(np.float32) / 255.0

    # Dark channel prior estimation
    # Larger kernel at higher strength for stronger haze removal
    k = int(round(9 + 6 * strength))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dark = np.min(img, axis=2)
    dark = cv2.erode(dark, kernel)

    # Estimate atmospheric light from top bright dark-channel pixels
    n = dark.size
    top_n = max(1, int(0.001 * n))
    flat_dark = dark.reshape(-1)
    flat_img = img.reshape(-1, 3)
    idx = np.argpartition(flat_dark, -top_n)[-top_n:]
    A = flat_img[idx].mean(axis=0)
    A = np.clip(A, 0.65, 1.0)

    # Transmission map
    norm = img / (A.reshape(1, 1, 3) + 1e-6)
    dark_norm = cv2.erode(np.min(norm, axis=2), kernel)
    omega = 0.72 + 0.2 * strength
    t = 1.0 - omega * dark_norm
    t_min = 0.5 - 0.25 * strength
    t = np.clip(t, t_min, 0.95)

    # Recover scene radiance
    J = (img - A.reshape(1, 1, 3)) / t[..., None] + A.reshape(1, 1, 3)
    J = np.clip(J, 0.0, 1.0)
    out = (J * 255.0).astype(np.uint8)

    # Gentle local contrast boost in LAB space
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.4 + 1.8 * strength, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)

    # Mild denoise to avoid crunchy artifacts
    out = cv2.bilateralFilter(out, d=5, sigmaColor=20 + 20 * (1.0 - strength), sigmaSpace=20 + 20 * (1.0 - strength))

    # Adaptive natural blend: more haze -> slightly stronger enhancement
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v_mean = np.mean(hsv[..., 2]) / 255.0
    s_mean = np.mean(hsv[..., 1]) / 255.0
    haze_score = np.clip(v_mean - s_mean, 0.0, 1.0)
    alpha_base = 0.15 + 0.35 * strength
    alpha_haze = 0.20 + 0.20 * strength
    alpha = float(np.clip(alpha_base + alpha_haze * haze_score, 0.15, 0.75))

    blended = cv2.addWeighted(img_rgb, 1.0 - alpha, out, alpha, 0)

    # Tiny gamma correction for natural midtones
    gamma = 0.98 - 0.10 * strength
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    blended = cv2.LUT(blended, lut)
    return np.clip(blended, 0, 255).astype(np.uint8)

def enhance_image(img_rgb, strength=0.5):
    """Enhance image using LP-GAN with adjustable fallback strength."""
    strength = float(np.clip(strength, 0.0, 1.0))
    h, w = img_rgb.shape[:2]
    img_small = cv2.resize(img_rgb, (256, 256))
    
    # Convert to tensor and normalize to [-1, 1]
    img_t = torch.from_numpy(img_small).float().permute(2, 0, 1).unsqueeze(0)
    img_t = img_t.to(device)
    # Normalize to [-1, 1]
    img_t = img_t / 127.5 - 1.0
    
    # Run GAN and sanity-check tensors
    def run_forward(x_t):
        with torch.no_grad():
            out = gan_model(x_t)
            if isinstance(out, (list, tuple)):
                t = out[0]
            else:
                t = out
            # ensure [B, C, H, W]
            if t.dim() == 4:
                return t[0]
            return t

    enhanced_t = run_forward(img_t)

    # Debug prints (helpful when investigating white/blank outputs)
    try:
        in_min, in_max = float(img_t.min()), float(img_t.max())
        out_min, out_max = float(enhanced_t.min()), float(enhanced_t.max())
        out_std = float(enhanced_t.std())
        print(f"[GAN] trial1 input range: {in_min:.4f}..{in_max:.4f}, output range: {out_min:.4f}..{out_max:.4f}, std={out_std:.6f}")
    except Exception:
        out_min, out_max, out_std = None, None, None

    # If output is saturated or nearly-constant, try alternate normalization
    saturated = False
    if out_min is not None:
        if (out_max - out_min) < 0.02 or out_min > 0.7 or out_max > 0.95:
            saturated = True

    used_fallback = False
    if saturated:
        # Try alternative normalization: [0,1] input
        print('[GAN] Output looks saturated; trying alternate normalization (img/255.0)')
        img_t_alt = torch.from_numpy(img_small).float().permute(2, 0, 1).unsqueeze(0).to(device)
        img_t_alt = img_t_alt / 255.0
        enhanced_t_alt = run_forward(img_t_alt)
        try:
            a_min, a_max = float(enhanced_t_alt.min()), float(enhanced_t_alt.max())
            a_std = float(enhanced_t_alt.std())
            print(f"[GAN] trial2 input range: {float(img_t_alt.min()):.4f}..{float(img_t_alt.max()):.4f}, output range: {a_min:.4f}..{a_max:.4f}, std={a_std:.6f}")
        except Exception:
            a_min, a_max = None, None

        # choose the better output (less saturated / higher std)
        if a_min is not None and ((a_max - a_min) > (out_max - out_min if out_min is not None else 0)):
            enhanced_t = enhanced_t_alt
            used_fallback = True

    # If still saturated, try swapping RGB <-> BGR channel order
    if (not used_fallback) and (out_min is not None) and ((out_max - out_min) < 0.02 or out_min > 0.7):
        print('[GAN] Trying channel swap (RGB->BGR) as fallback')
        img_swap = img_small[..., ::-1].copy()  # RGB->BGR
        img_t_swap = torch.from_numpy(img_swap).float().permute(2, 0, 1).unsqueeze(0).to(device)
        img_t_swap = img_t_swap / 127.5 - 1.0
        enhanced_t_swap = run_forward(img_t_swap)
        try:
            s_min, s_max = float(enhanced_t_swap.min()), float(enhanced_t_swap.max())
            s_std = float(enhanced_t_swap.std())
            print(f"[GAN] swap trial output range: {s_min:.4f}..{s_max:.4f}, std={s_std:.6f}")
        except Exception:
            s_min, s_max = None, None

        if s_min is not None and ((s_max - s_min) > (out_max - out_min if out_min is not None else 0)):
            enhanced_t = enhanced_t_swap
            used_fallback = True
            print('[GAN] Using channel-swap fallback')

    if used_fallback:
        print('[GAN] Using fallback normalization')

    # Convert from [-1, 1] to [0, 255]
    enhanced_t = (enhanced_t + 1.0) / 2.0
    enhanced_t = enhanced_t.clamp(0, 1)
    enhanced = (enhanced_t.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    # If GAN output is saturated/flat or too similar to input, use stronger classical dehazing fallback
    try:
        e_mean, e_std = float(enhanced.mean()), float(enhanced.std())
        diff_from_input = float(np.mean(np.abs(enhanced.astype(np.float32) - img_small.astype(np.float32))))
    except Exception:
        e_mean, e_std, diff_from_input = None, None, None

    if e_mean is not None and (e_mean > 240 or e_std < 5 or (diff_from_input is not None and diff_from_input < 6.0)):
        print(
            f"[ENHANCE] GAN output weak (mean={e_mean:.1f}, std={e_std:.2f}, diff={diff_from_input:.2f}); "
            "using classical dehaze fallback"
        )
        enhanced = classical_defog_enhance(img_small, strength=strength)
    
    # Resize back to original
    enhanced = cv2.resize(enhanced, (w, h))
    return enhanced

def detect(img_rgb):
    """Run YOLOv8 detection"""
    results = yolo_model.predict(img_rgb, verbose=False, conf=0.25)[0]
    dets = []
    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_name = results.names[int(box.cls[0])]
            dets.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)], 
                'conf': round(conf, 2), 
                'class': cls_name
            })
    return dets

def img_to_base64(img_rgb):
    """Convert image to base64"""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode()

def draw_boxes(img_rgb, dets):
    """Draw detection boxes on image"""
    img_copy = img_rgb.copy()
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d['class']} {d['conf']}"
        cv2.putText(img_copy, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img_to_base64(img_copy)

def safe_psnr(img1, img2):
    """Calculate PSNR safely"""
    try:
        val = calc_psnr(img1, img2, data_range=255)
        if val == float('inf') or val != val:  # NaN check
            return 50.0
        return round(float(val), 2)
    except:
        return 0.0

def safe_ssim(img1, img2):
    """Calculate SSIM safely"""
    try:
        val = calc_ssim(img1, img2, data_range=255)
        if val != val:  # NaN check
            return 0.0
        return round(float(val), 4)
    except:
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process():
    """Process image: enhance + detect"""
    try:
        file = request.files['image']
        strength_raw = request.form.get('strength', '50')
        try:
            strength = float(strength_raw) / 100.0
        except Exception:
            strength = 0.5
        strength = float(np.clip(strength, 0.0, 1.0))

        img_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Enhance
        enhanced = enhance_image(img_rgb, strength=strength)
        
        # Detect
        dets_orig = detect(img_rgb)
        dets_enh = detect(enhanced)
        
        # Metrics (safe calculation)
        psnr = safe_psnr(img_rgb, enhanced)
        ssim = safe_ssim(img_rgb, enhanced)
        
        # Draw boxes
        orig_boxes = draw_boxes(img_rgb, dets_orig)
        enh_boxes = draw_boxes(enhanced, dets_enh)
        
        return jsonify({
            'original_boxes': orig_boxes,
            'enhanced_boxes': enh_boxes,
            'dets_orig': dets_orig,
            'dets_enh': dets_enh,
            'psnr': psnr,
            'ssim': ssim,
            'count_orig': len(dets_orig),
            'count_enh': len(dets_enh),
            'strength': int(round(strength * 100)),
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n🚀 Server running at http://localhost:5000")
    app.run(debug=False, host='127.0.0.1', port=5000)