# Setup Guide (Simple)

This is the easiest way to run the project on Windows.

## First-Time Setup

1. Open terminal in the project folder.

```powershell
cd C:\Users\Shashank Reddy\fog-detection-app
```

2. Create virtual environment.

```powershell
python -m venv venv
```

3. Activate virtual environment.

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

CMD:

```bat
.\venv\Scripts\activate
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

4. Install dependencies.

```powershell
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -r requirements.txt
```

5. Start app.

```powershell
python app.py
```

6. Open browser:

http://127.0.0.1:5000

## How To Use

1. Upload image.
2. Move `Enhancement Style` slider.
3. Click `Process Image`.

Slider tip:
- `0-40`: natural look
- `40-70`: balanced
- `70-100`: strong enhancement

## Quick Run Next Time

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python app.py
```

CMD:

```bat
.\venv\Scripts\activate
python app.py
```

## Common Errors

### `ModuleNotFoundError` (torch/cv2)

Run inside activated venv:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -r requirements.txt
```

### `Set-ExecutionPolicy` is not recognized

You are in CMD. Use:

```bat
.\venv\Scripts\activate
```

### Port 5000 already in use

Stop old process with `Ctrl + C`, then run `python app.py` again.

