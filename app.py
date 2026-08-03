import os
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

# Download model on startup
os.makedirs("weights", exist_ok=True)
model_path = "weights/RealESRGAN_x4plus.pth"

if not os.path.exists(model_path):
    print("Downloading model...")
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

# Load model
print("Loading model...")
use_gpu = torch.cuda.is_available()

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
    half=use_gpu,
    gpu_id=0 if use_gpu else None
)

print(f"Model ready on: {'GPU' if use_gpu else 'CPU'}")

@app.get("/")
def home():
    return {
        "service": "PinAscend",
        "status": "running",
        "model": "RealESRGAN_x4plus",
        "gpu": use_gpu
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

        # Limit size for CPU (prevent timeout)
        max_size = 400
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
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
