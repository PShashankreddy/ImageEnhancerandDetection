# Fog Detection App

A Flask-based web app that enhances foggy images and runs YOLOv8 object detection on both the original and enhanced results.

## What It Does

- Upload a foggy or low-visibility image.
- Enhance the image with an LP-GAN-style restoration pipeline.
- Run YOLOv8 detection before and after enhancement.
- Show side-by-side detections, counts, and image quality metrics.

## Key Features

- Web UI for quick image upload and processing.
- Adjustable enhancement strength.
- Original vs enhanced detection comparison.
- PSNR and SSIM metrics for quality feedback.
- Bundled sample screenshots for a quick project preview.

## Sample Output Images

The screenshots below show the app in action from upload to detection results.

<table>
  <tr>
    <td align="center">
      <strong>App Preview</strong><br>
      <img src="image%20(1).png" alt="App preview screenshot" width="220">
    </td>
    <td align="center">
      <strong>Detection Summary</strong><br>
      <img src="image%20(2).png" alt="Detection summary screenshot" width="220">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Detection Comparison</strong><br>
      <img src="image%20(3).png" alt="Detection comparison screenshot" width="220">
    </td>
    <td align="center">
      <strong>Class Lists + Metrics</strong><br>
      <img src="image%20(4).png" alt="Class lists and metrics screenshot" width="220">
    </td>
  </tr>
</table>

## Quick Start (Windows)

### 1) Open terminal in the project folder

```powershell
cd C:\Users\Shashank Reddy\fog-detection-app
```

### 2) Create and activate a virtual environment

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

CMD:

```bat
python -m venv venv
.\venv\Scripts\activate
```

### 3) Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -r requirements.txt
```

### 4) Run the app

```powershell
python app.py
```

Open the app in your browser at http://127.0.0.1:5000.

## Usage

1. Upload an image.
2. Adjust the `Enhancement Style` slider.
   - Lower values keep the output more natural.
   - Higher values apply stronger fog removal.
3. Click `Process Image` to see the original and enhanced detections.

## Project Structure

- `app.py` - Flask backend, enhancement pipeline, and YOLOv8 inference.
- `templates/index.html` - Front-end UI for upload, preview, and results.
- `streamlit_app.py` - Streamlit prototype for the same workflow.
- `debug_outputs/` - Example outputs generated while testing the model.
- `checkpoints/` - Local enhancement model weights.

## More Setup Details

See `SETUP_GUIDE.md` for troubleshooting and Windows-specific notes.
