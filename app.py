import os
import sys
import subprocess

# Fix basicsr degradations import bug BEFORE importing
def fix_basicsr():
    try:
        import basicsr.data.degradations as deg
        content = open(deg.__file__).read()
        if 'functional_tensor' in content:
            new_content = content.replace(
                'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
                'from torchvision.transforms.functional import rgb_to_grayscale'
            )
            with open(deg.__file__, 'w') as f:
                f.write(new_content)
            print("Fixed basicsr!")
    except Exception as e:
        print(f"Fix skipped: {e}")

fix_basicsr()

import torch
import cv2
import gc
import numpy as np
import urllib.request
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from io import BytesIO

app = FastAPI(title="PinAscend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Original-Size", "X-Upscaled-Size", "X-Scale"]
)

# Download model
os.makedirs("weights", exist_ok=True)
model_path = "weights/RealESRGAN_x4plus.pth"

if not os.path.exists(model_path):
    print("Downloading model...")
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

print("Loading model...")
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=200,
    tile_pad=10,
    pre_pad=0,
    half=False,
    gpu_id=None
)

print("Model ready!")

@app.get("/")
def home():
    return {
        "service": "PinAscend",
        "status": "running",
        "model": "RealESRGAN_x4plus",
        "gpu": torch.cuda.is_available()
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upscale/")
async def upscale_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image"}
            )

        h, w = img.shape[:2]
        if max(h, w) > 400:
            scale = 400 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

        orig_h, orig_w = img.shape[:2]
        output, _ = upsampler.enhance(img, outscale=4)
        new_h, new_w = output.shape[:2]

        _, img_encoded = cv2.imencode(".png", output)
        img_bytes = BytesIO(img_encoded.tobytes())

        gc.collect()

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={
                "X-Original-Size": f"{orig_w}x{orig_h}",
                "X-Upscaled-Size": f"{new_w}x{new_h}",
                "X-Scale": "4x",
                "Access-Control-Expose-Headers": "X-Original-Size, X-Upscaled-Size, X-Scale"
            }
        )

    except Exception as e:
        gc.collect()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
