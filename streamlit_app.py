# streamlit_app.py — Ultra-Simplified Version

import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Image Enhancement", page_icon="🌫️", layout="wide")

# Header
st.markdown("""
<div style="text-align:center; padding:2rem; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
border-radius:10px; color:white; margin-bottom:2rem;">
    <h1>🌫️ Image Enhancement & Detection</h1>
    <p>AI-powered visibility restoration in adverse weather</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# GAN MODEL
# ──────────────────────────────────────────────────────────────────────────

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
        self.block = nn.Sequential(DWSConvBlock(ch, ch), nn.Conv2d(ch, ch, 1, bias=False), nn.InstanceNorm2d(ch, affine=True))
    def forward(self, x): return x + self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), DWSConvBlock(in_ch, out_ch))
    def forward(self, x): return self.block(x)

class LPGANGenerator(nn.Module):
    def __init__(self, base_ch=32, n_res=6):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, base_ch, 7, 1, 3, bias=False), nn.InstanceNorm2d(base_ch, affine=True), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = DWSConvBlock(base_ch, base_ch*2, stride=2)
        self.enc3 = DWSConvBlock(base_ch*2, base_ch*4, stride=2)
        self.enc4 = DWSConvBlock(base_ch*4, base_ch*8, stride=2)
        self.bottleneck = nn.Sequential(*[ResBlock(base_ch*8) for _ in range(n_res)])
        self.dec4 = UpBlock(base_ch*8 + base_ch*8, base_ch*4)
        self.dec3 = UpBlock(base_ch*4 + base_ch*4, base_ch*2)
        self.dec2 = UpBlock(base_ch*2 + base_ch*2, base_ch)
        self.out = nn.Sequential(nn.Conv2d(base_ch + base_ch, base_ch, 3, 1, 1, bias=False), nn.InstanceNorm2d(base_ch, affine=True), nn.ReLU(inplace=True), nn.Conv2d(base_ch, 3, 7, 1, 3), nn.Tanh())
    
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

# ──────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ──────────────────────────────────────────────────────────────────────────

try:
    device = 'cpu'  # Use CPU to avoid CUDA issues
    
    st.info("⏳ Loading models... (first time may take 30 seconds)")
    
    yolo_model = YOLO('yolov8s.pt')
    gan_model = LPGANGenerator(base_ch=32, n_res=6).to(device)
    gan_model.eval()
    
    st.success("✅ Models loaded!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────

def enhance_image(img_rgb):
    h, w = img_rgb.shape[:2]
    img_small = cv2.resize(img_rgb, (256, 256))
    img_t = torch.from_numpy(img_small).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    
    with torch.no_grad():
        enhanced_t = gan_model(img_t)[0]
    
    enhanced_t = (enhanced_t + 1.0) / 2.0
    enhanced = (enhanced_t.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    enhanced = cv2.resize(enhanced, (w, h))
    return enhanced

def detect(img):
    results = yolo_model.predict(img, verbose=False, conf=0.25)[0]
    dets = []
    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_name = results.names[int(box.cls[0])]
            dets.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'class': cls_name})
    return dets

def draw_boxes(img, dets, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"{d['class']} {d['conf']:.2f}", color='lime', fontsize=9, weight='bold', 
                bbox=dict(facecolor='black', alpha=0.7))
    ax.axis('off')
    ax.set_title(title, fontweight='bold', fontsize=12)
    return fig

# ──────────────────────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded = st.file_uploader("Choose image (PNG, JPG)", type=['png', 'jpg', 'jpeg'])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    with col2:
        st.subheader("👀 Preview")
        st.image(img_rgb, use_column_width=True)
    
    st.markdown("---")
    
    if st.button("🚀 Process Image", use_container_width=True):
        with st.spinner("Processing..."):
            # Enhance
            enhanced = enhance_image(img_rgb)
            
            # Detect
            dets_orig = detect(img_rgb)
            dets_enh = detect(enhanced)
            
            # Metrics
            psnr = calc_psnr(img_rgb, enhanced, data_range=255)
            ssim = calc_ssim(img_rgb, enhanced, channel_axis=2, data_range=255)
        
        st.success("✅ Done!")
        
        # Results
        st.markdown("## Results")
        
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader(f"Original ({len(dets_orig)} objects)")
            fig1 = draw_boxes(img_rgb, dets_orig, "Original")
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col_r:
            st.subheader(f"Enhanced ({len(dets_enh)} objects)")
            fig2 = draw_boxes(enhanced, dets_enh, "Enhanced")
            st.pyplot(fig2)
            plt.close(fig2)
        
        # Metrics
        st.markdown("---")
        st.markdown("## Metrics")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("PSNR (dB)", f"{psnr:.2f}")
        with m2:
            st.metric("SSIM", f"{ssim:.4f}")
        with m3:
            improvement = len(dets_enh) - len(dets_orig)
            st.metric("Detection Change", f"{improvement:+d}", delta_color="off")
        
        # Detections
        st.markdown("---")
        st.markdown("## Detections")
        
        d1, d2 = st.columns(2)
        with d1:
            st.subheader("Original")
            if dets_orig:
                for d in dets_orig:
                    st.write(f"🎯 {d['class']}: {d['conf']*100:.1f}%")
            else:
                st.info("No objects")
        
        with d2:
            st.subheader("Enhanced")
            if dets_enh:
                for d in dets_enh:
                    st.write(f"🎯 {d['class']}: {d['conf']*100:.1f}%")
            else:
                st.info("No objects")

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>AI-Based Image Enhancement & Object Detection | Minor Project</p>", unsafe_allow_html=True)