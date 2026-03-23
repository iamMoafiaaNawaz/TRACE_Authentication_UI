# TRACE Skin Lesion Detection API

Dual-model dermoscopy analysis service — YOLOv11x localisation + ConvNeXt-Base classification

> **Disclaimer:** This API is for research assistance only. It is **not** a
> substitute for clinical diagnosis. All detections and classifications must be
> reviewed by a qualified dermatologist before any clinical decision is made.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Endpoint Reference](#endpoint-reference)
   - [GET /health](#get-health)
   - [POST /detect](#post-detect)
   - [POST /detect/batch](#post-detectbatch)
   - [POST /classify](#post-classify)
   - [POST /classify/batch](#post-classifybatch)
4. [Request & Response Schemas](#request--response-schemas)
5. [Environment Variables](#environment-variables)
6. [Class Descriptions](#class-descriptions)
7. [Error Codes](#error-codes)
8. [Production Deployment](#production-deployment)
9. [Python Client Example](#python-client-example)

---

## Overview

The TRACE Skin Lesion Detection API wraps two trained models and exposes them
as a unified RESTful HTTP service built with **FastAPI** + **Uvicorn**:

- **YOLOv11x** — object detection for bounding-box localisation of skin lesions
- **ConvNeXt-Base** — 4-class classification of dermoscopy images

| Property            | YOLO (detect)                                 | ConvNeXt (classify)                                          |
|---------------------|-----------------------------------------------|--------------------------------------------------------------|
| Model               | YOLOv11x (Exp8)                               | ConvNeXt-Base (progressive fine-tuned)                       |
| Task                | Object detection                              | 4-class classification                                       |
| Weights             | `weights/yolo/yolov11x_best.pt`               | `weights/convnext/best_convnext_checkpoint.pth`              |
| Input resolution    | 640×640 (auto-padded)                         | 224×224 (ResizePad aspect-preserving)                        |
| Output              | Bounding boxes                                | Softmax probabilities                                        |
| Default conf/thresh | 0.25 conf / 0.70 IoU                          | —                                                            |
| Framework           | FastAPI 0.110+ / Pydantic v2                  | FastAPI 0.110+ / Pydantic v2                                 |

The service exposes five endpoints:

- `GET  /health`           — liveness probe and model status for both models
- `POST /detect`           — single-image lesion detection (YOLO)
- `POST /detect/batch`     — multi-image batch detection (YOLO, up to 32 images)
- `POST /classify`         — single-image classification (ConvNeXt)
- `POST /classify/batch`   — multi-image batch classification (ConvNeXt, up to 32 images)

Interactive API docs are available at `http://localhost:8000/docs` (Swagger UI)
and `http://localhost:8000/redoc` (ReDoc) once the server is running.

---

## Quick Start

### 1. Install dependencies

```bash
pip install fastapi uvicorn[standard] ultralytics pillow numpy pydantic torch torchvision
```

### 2. Place model weights

Copy your trained checkpoints to:

```
weights/yolo/yolov11x_best.pt
weights/convnext/best_convnext_checkpoint.pth
```

Both `weights/yolo/` and `weights/convnext/` directories are tracked in the
repository via `.gitkeep`. The actual model files are excluded from git due to
their size.

### 3. Start the server

**Development (auto-reload on code changes):**

```bash
python main_api.py --reload
```

**Explicit uvicorn invocation:**

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Production (multiple workers):**

```bash
python main_api.py --host 0.0.0.0 --port 8000 --workers 4
```

The server starts at `http://0.0.0.0:8000`.

### 4. Test with curl

**Health check:**

```bash
curl http://localhost:8000/health
```

**Single image detection (YOLO):**

```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@/path/to/dermoscopy.jpg" \
  -F "conf=0.30"
```

**Single image classification (ConvNeXt):**

```bash
curl -X POST http://localhost:8000/classify \
  -F "image=@/path/to/dermoscopy.jpg"
```

**Batch detection:**

```bash
curl -X POST http://localhost:8000/detect/batch \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "conf=0.25" \
  -F "iou=0.7"
```

**Batch classification:**

```bash
curl -X POST http://localhost:8000/classify/batch \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg"
```

---

## Endpoint Reference

### GET /health

Returns the current liveness status of the API and whether both model
checkpoints have been successfully loaded into memory.

**No request body required.**

#### Response — 200 OK

```json
{
  "status": "ok",
  "yolo_loaded": true,
  "yolo_path": "/absolute/path/to/weights/yolo/yolov11x_best.pt",
  "convnext_loaded": true,
  "convnext_path": "/absolute/path/to/weights/convnext/best_convnext_checkpoint.pth",
  "classes": ["BCC", "BKL", "MEL", "NV"],
  "device": "cpu",
  "version": "1.1.0"
}
```

| Field             | Type    | Description                                          |
|-------------------|---------|------------------------------------------------------|
| `status`          | string  | Always `"ok"` when the server is reachable           |
| `yolo_loaded`     | boolean | `true` if YOLO weights were loaded at startup        |
| `yolo_path`       | string  | Absolute path to the YOLO weights file               |
| `convnext_loaded` | boolean | `true` if ConvNeXt checkpoint was loaded at startup  |
| `convnext_path`   | string  | Absolute path to the ConvNeXt checkpoint file        |
| `classes`         | array   | Ordered class list matching dataset.yaml             |
| `device`          | string  | Inference device (`cpu`, `cuda:0`, etc.)             |
| `version`         | string  | API semantic version                                 |

**Note:** Either `yolo_loaded` or `convnext_loaded` can be `false` if a weights
file was absent at startup. In that case the corresponding endpoint will return
HTTP 500 until weights are placed and the server is restarted.

---

### POST /detect

Detect skin lesions in a single dermoscopy image using YOLOv11x.

**Content-Type:** `multipart/form-data`

#### Request fields

| Field   | Type  | Required | Default | Description                               |
|---------|-------|----------|---------|-------------------------------------------|
| `image` | file  | Yes      | —       | Image file (JPEG, PNG, BMP, TIFF)         |
| `conf`  | float | No       | 0.25    | Confidence threshold (0.01 – 1.0)         |
| `iou`   | float | No       | 0.70    | IoU NMS threshold (0.1 – 1.0)             |

#### Response — 200 OK

```json
{
  "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "model": "YOLOv11x",
  "image_width": 1024,
  "image_height": 768,
  "num_detections": 2,
  "detections": [
    {
      "detection_id": 0,
      "class_id": 2,
      "class_name": "MEL",
      "confidence": 0.8741,
      "box": {
        "x1": 112.3,
        "y1": 88.7,
        "x2": 340.1,
        "y2": 295.4,
        "cx_norm": 0.2208,
        "cy_norm": 0.2506,
        "w_norm":  0.2224,
        "h_norm":  0.2694
      },
      "class_info": {
        "full_name": "Melanoma",
        "risk": "Critical",
        "icd10": "C43.9",
        "action": "URGENT: Immediate oncology referral"
      }
    },
    {
      "detection_id": 1,
      "class_id": 3,
      "class_name": "NV",
      "confidence": 0.6123,
      "box": {
        "x1": 600.0,
        "y1": 400.0,
        "x2": 750.0,
        "y2": 560.0,
        "cx_norm": 0.6592,
        "cy_norm": 0.6250,
        "w_norm":  0.1465,
        "h_norm":  0.2083
      },
      "class_info": {
        "full_name": "Melanocytic Nevus (Benign Mole)",
        "risk": "Low",
        "icd10": "D22.9",
        "action": "Routine monitoring; annual skin check recommended"
      }
    }
  ],
  "conf_threshold": 0.25,
  "iou_threshold": 0.7,
  "inference_time_ms": 143.27,
  "warning": "AI output is for research assistance only. Not a substitute for clinical diagnosis."
}
```

---

### POST /detect/batch

Detect lesions in multiple images in a single request using YOLOv11x.

**Content-Type:** `multipart/form-data`

**Limit:** Maximum 32 images per request.

#### Request fields

| Field    | Type        | Required | Default | Description                               |
|----------|-------------|----------|---------|-------------------------------------------|
| `images` | file (list) | Yes      | —       | One or more image files                   |
| `conf`   | float       | No       | 0.25    | Confidence threshold applied to all       |
| `iou`    | float       | No       | 0.70    | IoU NMS threshold applied to all          |

#### Response — 200 OK

```json
{
  "total_images": 2,
  "results": [
    {
      "request_id": "abc123...",
      "model": "YOLOv11x",
      "image_width": 1024,
      "image_height": 768,
      "num_detections": 1,
      "detections": [ "..." ],
      "conf_threshold": 0.25,
      "iou_threshold": 0.7,
      "inference_time_ms": 138.50,
      "warning": "AI output is for research assistance only. Not a substitute for clinical diagnosis."
    },
    {
      "request_id": "def456...",
      "...": "..."
    }
  ]
}
```

Each element of `results` has the same schema as a single `/detect` response.
Images are processed sequentially; `inference_time_ms` reflects per-image time.

---

### POST /classify

Classify a single dermoscopy image into one of four lesion categories using ConvNeXt-Base.

**Content-Type:** `multipart/form-data`

**Weights loaded from:** `weights/convnext/best_convnext_checkpoint.pth`

#### Request fields

| Field   | Type | Required | Description              |
|---------|------|----------|--------------------------|
| `image` | file | Yes      | Image file (JPEG, PNG, BMP, TIFF) |

#### Response — 200 OK

```json
{
  "request_id": "uuid4",
  "model": "ConvNeXt-Base",
  "image_width": 224,
  "image_height": 224,
  "predicted_class": "MEL",
  "predicted_class_id": 2,
  "confidence": 0.9005,
  "probabilities": [
    {"class_name": "BCC", "class_id": 0, "probability": 0.0258},
    {"class_name": "BKL", "class_id": 1, "probability": 0.0593},
    {"class_name": "MEL", "class_id": 2, "probability": 0.9005},
    {"class_name": "NV",  "class_id": 3, "probability": 0.0143}
  ],
  "class_info": {
    "full_name": "Melanoma",
    "risk": "Critical",
    "icd10": "C43.9",
    "action": "URGENT: Immediate oncology referral"
  },
  "inference_time_ms": 556.1,
  "warning": "AI output is for research assistance only. Not a substitute for clinical diagnosis."
}
```

---

### POST /classify/batch

Classify multiple dermoscopy images in a single request using ConvNeXt-Base.

**Content-Type:** `multipart/form-data`

**Limit:** Maximum 32 images per request.

#### Request fields

| Field    | Type        | Required | Description                    |
|----------|-------------|----------|--------------------------------|
| `images` | file (list) | Yes      | One or more image files        |

#### Response — 200 OK

```json
{
  "total_images": 2,
  "results": [
    {
      "request_id": "uuid4",
      "model": "ConvNeXt-Base",
      "image_width": 224,
      "image_height": 224,
      "predicted_class": "MEL",
      "predicted_class_id": 2,
      "confidence": 0.9005,
      "probabilities": [ "..." ],
      "class_info": { "..." : "..." },
      "inference_time_ms": 556.1,
      "warning": "AI output is for research assistance only. Not a substitute for clinical diagnosis."
    },
    {
      "request_id": "uuid4",
      "...": "..."
    }
  ]
}
```

Each element of `results` has the same schema as a single `/classify` response.

---

## Request & Response Schemas

### BoundingBox

Both pixel-space absolute coordinates and image-normalised (0–1) values are
returned for every detection, making it easy to draw overlays regardless of
display resolution.

| Field     | Type  | Description                              |
|-----------|-------|------------------------------------------|
| `x1`      | float | Left edge in pixels                      |
| `y1`      | float | Top edge in pixels                       |
| `x2`      | float | Right edge in pixels                     |
| `y2`      | float | Bottom edge in pixels                    |
| `cx_norm` | float | Normalised centre-x (0–1)                |
| `cy_norm` | float | Normalised centre-y (0–1)                |
| `w_norm`  | float | Normalised width (0–1)                   |
| `h_norm`  | float | Normalised height (0–1)                  |

### Detection

| Field          | Type        | Description                                |
|----------------|-------------|--------------------------------------------|
| `detection_id` | int         | Zero-based index within this image         |
| `class_id`     | int         | Numeric class (0=BCC, 1=BKL, 2=MEL, 3=NV) |
| `class_name`   | string      | Short class label                          |
| `confidence`   | float       | Model confidence score (0–1)               |
| `box`          | BoundingBox | Bounding box details                       |
| `class_info`   | ClassInfo   | Clinical metadata for this class           |

### ClassProbability

| Field         | Type   | Description                          |
|---------------|--------|--------------------------------------|
| `class_name`  | string | Short class label (BCC/BKL/MEL/NV)   |
| `class_id`    | int    | Numeric class index                  |
| `probability` | float  | Softmax probability (0–1)            |

### ClassInfo

| Field       | Type   | Description                              |
|-------------|--------|------------------------------------------|
| `full_name` | string | Full clinical name of the lesion type    |
| `risk`      | string | Risk level: Low / High / Critical        |
| `icd10`     | string | ICD-10-CM code                           |
| `action`    | string | Recommended clinical action              |

### DetectionResponse

| Field               | Type           | Description                          |
|---------------------|----------------|--------------------------------------|
| `request_id`        | string (UUID4) | Unique ID for traceability           |
| `model`             | string         | Model identifier (`YOLOv11x`)        |
| `image_width`       | int            | Original image width in pixels       |
| `image_height`      | int            | Original image height in pixels      |
| `num_detections`    | int            | Count of lesions found               |
| `detections`        | list           | Array of Detection objects           |
| `conf_threshold`    | float          | Confidence threshold used            |
| `iou_threshold`     | float          | IoU threshold used                   |
| `inference_time_ms` | float          | Wall-clock inference time            |
| `warning`           | string         | Clinical disclaimer                  |

### ClassificationResponse

| Field                | Type           | Description                          |
|----------------------|----------------|--------------------------------------|
| `request_id`         | string (UUID4) | Unique ID for traceability           |
| `model`              | string         | Model identifier (`ConvNeXt-Base`)   |
| `image_width`        | int            | Image width after preprocessing      |
| `image_height`       | int            | Image height after preprocessing     |
| `predicted_class`    | string         | Top-1 class label                    |
| `predicted_class_id` | int            | Top-1 class index                    |
| `confidence`         | float          | Top-1 softmax probability            |
| `probabilities`      | list           | Full softmax distribution (4 items)  |
| `class_info`         | ClassInfo      | Clinical metadata for predicted class|
| `inference_time_ms`  | float          | Wall-clock inference time            |
| `warning`            | string         | Clinical disclaimer                  |

---

## Environment Variables

All inference settings can be overridden without editing code, making the
same Docker image deployable across environments.

| Variable               | Default                                             | Description                                   |
|------------------------|-----------------------------------------------------|-----------------------------------------------|
| `YOLO_MODEL_PATH`      | `weights/yolo/yolov11x_best.pt`                     | Absolute or relative path to YOLO weights     |
| `YOLO_IMGSZ`           | `640`                                               | YOLO inference image size (pixels, square)    |
| `YOLO_CONF`            | `0.25`                                              | Default confidence threshold                  |
| `YOLO_IOU`             | `0.7`                                               | Default IoU NMS threshold                     |
| `YOLO_MAX_DET`         | `300`                                               | Maximum detections per image                  |
| `YOLO_DEVICE`          | `cpu`                                               | PyTorch device string (`cpu`, `cuda:0`, etc.) |
| `CONVNEXT_MODEL_PATH`  | `weights/convnext/best_convnext_checkpoint.pth`     | Path to ConvNeXt .pth checkpoint              |
| `CONVNEXT_IMGSZ`       | `224`                                               | ConvNeXt inference image size                 |

**Example — override to use GPU and custom weight paths:**

```bash
export YOLO_DEVICE=cuda:0
export YOLO_MODEL_PATH=/mnt/models/yolov11x_best.pt
export CONVNEXT_MODEL_PATH=/mnt/models/best_convnext_checkpoint.pth
python main_api.py --host 0.0.0.0 --port 8000
```

**Per-request overrides:** `conf` and `iou` can also be set individually
per `/detect` request via the form fields, overriding the environment-level
defaults. Classification has no per-request threshold parameters.

---

## Class Descriptions

Both models detect/classify the same four dermoscopy lesion classes in the
order they appear in `dataset.yaml`. Class index order must match what was
used during training.

### BCC — Basal Cell Carcinoma

| Property  | Value                              |
|-----------|------------------------------------|
| Class ID  | 0                                  |
| ICD-10    | C44.91                             |
| Risk      | **High**                           |
| Action    | Urgent dermatology referral required |

The most common form of skin cancer. Typically appears as a pearly or waxy
bump, flat flesh-coloured lesion, or bleeding sore. Rarely metastasises but
requires prompt treatment to prevent local tissue destruction.

### BKL — Benign Keratosis-like Lesion

| Property  | Value                                            |
|-----------|--------------------------------------------------|
| Class ID  | 1                                                |
| ICD-10    | L82.1                                            |
| Risk      | **Low**                                          |
| Action    | Monitor; reassess if growth or change observed   |

Encompasses seborrhoeic keratoses and solar lentigines. Benign, non-cancerous
growths. Monitoring is typically sufficient unless features atypical of BKL
emerge.

### MEL — Melanoma

| Property  | Value                                    |
|-----------|------------------------------------------|
| Class ID  | 2                                        |
| ICD-10    | C43.9                                    |
| Risk      | **Critical**                             |
| Action    | URGENT: Immediate oncology referral      |

The most dangerous form of skin cancer. Arises from melanocytes and can
metastasise rapidly if untreated. Any MEL detection should trigger immediate
clinical review regardless of confidence score.

### NV — Melanocytic Nevus (Benign Mole)

| Property  | Value                                                  |
|-----------|--------------------------------------------------------|
| Class ID  | 3                                                      |
| ICD-10    | D22.9                                                  |
| Risk      | **Low**                                                |
| Action    | Routine monitoring; annual skin check recommended      |

Common benign melanocytic lesions. Low malignant potential but require annual
monitoring for changes in size, shape, or colour (ABCDE criteria).

---

## Error Codes

| HTTP Status | Condition                                                                         |
|-------------|-----------------------------------------------------------------------------------|
| 200         | Success — results returned (detections may be an empty list)                      |
| 400         | Bad request — e.g. batch size > 32                                                |
| 422         | Unprocessable entity — invalid file type, corrupt image, or form validation error |
| 500         | Internal server error — model not loaded, inference crash, or unexpected exception |

### Error response body (422 example)

```json
{
  "detail": "Cannot decode image: cannot identify image file <_io.BytesIO object>"
}
```

### Error response body (400 example)

```json
{
  "detail": "Batch size limited to 32 images per request."
}
```

### Error response body (500 example)

When the model weights file is missing and a request is made:

```json
{
  "detail": "Internal Server Error"
}
```

Server logs will contain the full traceback including the `FileNotFoundError`
with the expected weights path.

---

## Production Deployment

### Gunicorn + Uvicorn workers

For production deployments requiring process isolation:

```bash
pip install gunicorn
gunicorn src.api.app:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

**Note:** Do not use `--reload` in production.

### Docker

A minimal `Dockerfile` for the API:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Mount weights at runtime — do not bake into image
ENV YOLO_MODEL_PATH=/weights/yolo/yolov11x_best.pt
ENV CONVNEXT_MODEL_PATH=/weights/convnext/best_convnext_checkpoint.pth

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

Run with both weight directories mounted:

```bash
docker build -t trace-api .
docker run -p 8000:8000 \
  -v /path/to/weights/yolo:/weights/yolo:ro \
  -v /path/to/weights/convnext:/weights/convnext:ro \
  -e YOLO_MODEL_PATH=/weights/yolo/yolov11x_best.pt \
  -e CONVNEXT_MODEL_PATH=/weights/convnext/best_convnext_checkpoint.pth \
  -e YOLO_DEVICE=cpu \
  trace-api
```

### GPU inference

Install the CUDA-enabled PyTorch and ultralytics builds, then:

```bash
export YOLO_DEVICE=cuda:0
python main_api.py --host 0.0.0.0 --port 8000
```

### CORS restriction

`app.py` defaults to `allow_origins=["*"]` for research/dev convenience.
Restrict this in production by editing `src/api/app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

### Reverse proxy (nginx example)

```nginx
upstream trace_api {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl;
    server_name api.your-domain.com;

    client_max_body_size 50M;  # allow large image uploads

    location / {
        proxy_pass         http://trace_api;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

---

## Python Client Example

A complete Python example using the `requests` library:

```python
"""
Example: call the TRACE Skin Lesion Detection API from Python.

Install: pip install requests
"""
import requests
from pathlib import Path


BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------
def check_health(base_url: str = BASE_URL) -> dict:
    resp = requests.get(f"{base_url}/health", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    print(f"API status       : {data['status']}")
    print(f"YOLO loaded      : {data['yolo_loaded']}")
    print(f"ConvNeXt loaded  : {data['convnext_loaded']}")
    print(f"Classes          : {data['classes']}")
    return data


# ---------------------------------------------------------------------------
# 2. Single image detection (YOLO)
# ---------------------------------------------------------------------------
def detect_single(
    image_path: str | Path,
    conf: float = 0.25,
    iou: float = 0.7,
    base_url: str = BASE_URL,
) -> dict:
    image_path = Path(image_path)
    with open(image_path, "rb") as fh:
        resp = requests.post(
            f"{base_url}/detect",
            files={"image": (image_path.name, fh, "image/jpeg")},
            data={"conf": conf, "iou": iou},
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# 3. Batch detection (YOLO)
# ---------------------------------------------------------------------------
def detect_batch(
    image_paths: list[str | Path],
    conf: float = 0.25,
    iou: float = 0.7,
    base_url: str = BASE_URL,
) -> dict:
    files = []
    handles = []
    for p in image_paths:
        p = Path(p)
        fh = open(p, "rb")
        handles.append(fh)
        files.append(("images", (p.name, fh, "image/jpeg")))

    try:
        resp = requests.post(
            f"{base_url}/detect/batch",
            files=files,
            data={"conf": conf, "iou": iou},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    finally:
        for fh in handles:
            fh.close()


# ---------------------------------------------------------------------------
# 4. Pretty-print detections
# ---------------------------------------------------------------------------
def print_detections(result: dict) -> None:
    print(f"\nRequest ID   : {result['request_id']}")
    print(f"Image size   : {result['image_width']} x {result['image_height']}")
    print(f"Detections   : {result['num_detections']}")
    print(f"Inference    : {result['inference_time_ms']:.1f} ms")

    if result["num_detections"] == 0:
        print("  (no lesions detected above confidence threshold)")
        return

    for det in result["detections"]:
        box = det["box"]
        info = det["class_info"]
        print(
            f"\n  [{det['detection_id']}] {det['class_name']} "
            f"({info['full_name']}) — conf={det['confidence']:.3f}"
        )
        print(f"      Risk   : {info['risk']}")
        print(f"      ICD-10 : {info['icd10']}")
        print(f"      Action : {info['action']}")
        print(
            f"      Box    : ({box['x1']}, {box['y1']}) -> "
            f"({box['x2']}, {box['y2']})  "
            f"[cx={box['cx_norm']:.3f}, cy={box['cy_norm']:.3f}, "
            f"w={box['w_norm']:.3f}, h={box['h_norm']:.3f}]"
        )


# ---------------------------------------------------------------------------
# 5. Single image classification (ConvNeXt)
# ---------------------------------------------------------------------------
def classify_single(image_path, base_url=BASE_URL):
    image_path = Path(image_path)
    with open(image_path, "rb") as fh:
        resp = requests.post(
            f"{base_url}/classify",
            files={"image": (image_path.name, fh, "image/jpeg")},
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# 6. Batch classification (ConvNeXt)
# ---------------------------------------------------------------------------
def classify_batch(image_paths, base_url=BASE_URL):
    files = []
    handles = []
    for p in image_paths:
        p = Path(p)
        fh = open(p, "rb")
        handles.append(fh)
        files.append(("images", (p.name, fh, "image/jpeg")))
    try:
        resp = requests.post(f"{base_url}/classify/batch", files=files, timeout=120)
        resp.raise_for_status()
        return resp.json()
    finally:
        for fh in handles:
            fh.close()


# ---------------------------------------------------------------------------
# 7. Pretty-print classification
# ---------------------------------------------------------------------------
def print_classification(result):
    print(f"\nRequest ID   : {result['request_id']}")
    print(f"Image size   : {result['image_width']} x {result['image_height']}")
    print(f"Predicted    : {result['predicted_class']}  (conf={result['confidence']:.3f})")
    print(f"Risk         : {result['class_info']['risk']}")
    print(f"Action       : {result['class_info']['action']}")
    print(f"Inference    : {result['inference_time_ms']:.1f} ms")
    print("Full distribution:")
    for p in result["probabilities"]:
        bar = "#" * int(p["probability"] * 30)
        print(f"  {p['class_name']:4s}: {p['probability']:.4f} {bar}")


# ---------------------------------------------------------------------------
# 8. Main demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Health check
    check_health()

    # Single image (pass path as first CLI arg)
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        print(f"\n--- Single detection: {image_file} ---")
        result = detect_single(image_file, conf=0.25)
        print_detections(result)

        print(f"\n--- Single classification: {image_file} ---")
        cls_result = classify_single(image_file)
        print_classification(cls_result)

    # Batch detection and classification (all remaining args)
    if len(sys.argv) > 2:
        batch_paths = sys.argv[1:]
        print(f"\n--- Batch detection: {len(batch_paths)} images ---")
        batch_result = detect_batch(batch_paths, conf=0.25)
        print(f"Total images processed: {batch_result['total_images']}")
        for r in batch_result["results"]:
            print_detections(r)

        print(f"\n--- Batch classification: {len(batch_paths)} images ---")
        cls_batch_result = classify_batch(batch_paths)
        print(f"Total images processed: {cls_batch_result['total_images']}")
        for r in cls_batch_result["results"]:
            print_classification(r)
```

**Run the example:**

```bash
# Single image
python client_example.py /path/to/lesion.jpg

# Batch
python client_example.py img1.jpg img2.jpg img3.png
```

---

*Generated for TRACE Skin Cancer Detection — Dual-model API v1.1.0 (YOLOv11x Exp8 + ConvNeXt-Base classifier).*
