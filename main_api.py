# -*- coding: utf-8 -*-
"""
main_api.py
===========
Uvicorn entry point for the TRACE Skin Lesion Detection API.

Usage
-----
    # Development (auto-reload):
    python main_api.py

    # Production:
    python main_api.py --host 0.0.0.0 --port 8000 --workers 2

    # Or directly with uvicorn:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Environment overrides
---------------------
    YOLO_MODEL_PATH   path to best.pt (default: weights/yolo/yolov11x_best.pt)
    YOLO_CONF         confidence threshold (default: 0.25)
    YOLO_IOU          IoU NMS threshold   (default: 0.7)
    YOLO_DEVICE       device string       (default: cpu)
    YOLO_IMGSZ        inference image size (default: 640)
"""

import argparse
import uvicorn


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog        = "main_api.py",
        description = "TRACE Skin Lesion Detection API server",
    )
    p.add_argument("--host",    default="0.0.0.0")
    p.add_argument("--port",    type=int, default=8000)
    p.add_argument("--workers", type=int, default=1,
                   help="Uvicorn worker processes (set >1 for production)")
    p.add_argument("--reload",  action="store_true",
                   help="Enable auto-reload (development only)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    uvicorn.run(
        "src.api.app:app",
        host    = args.host,
        port    = args.port,
        workers = args.workers,
        reload  = args.reload,
    )
