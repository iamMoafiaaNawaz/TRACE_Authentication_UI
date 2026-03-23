import os
import uuid
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class HairRemovalService:
    """Loads hair-removal model once and exposes reusable processing methods."""

    def __init__(self, model_path: str, outputs_dir: str, allow_fallback: bool = True):
        self.model_path = model_path
        self.outputs_dir = outputs_dir
        self.allow_fallback = allow_fallback
        self.model = None
        self.model_backend = "opencv_dullrazor"
        self.model_error: Optional[str] = None

        os.makedirs(self.outputs_dir, exist_ok=True)

    def load_model(self) -> None:
        """Load model into memory once (called on app startup)."""
        if not os.path.exists(self.model_path):
            self.model = None
            self.model_backend = "opencv_dullrazor_fallback"
            self.model_error = f"Model not found: {self.model_path}"
            return

        try:
            from tensorflow.keras.models import load_model  # type: ignore

            self.model = load_model(self.model_path, compile=False)
            self.model_backend = "keras_h5"
            self.model_error = None
        except Exception as exc:
            self.model = None
            self.model_backend = "opencv_dullrazor_fallback"
            self.model_error = str(exc)

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image bytes")
        return image

    def _fallback_dullrazor(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask_bin = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        mask_bin = cv2.dilate(mask_bin, np.ones((3, 3), np.uint8), iterations=1)
        processed = cv2.inpaint(image_bgr, mask_bin, 3, cv2.INPAINT_TELEA)
        return processed, mask_bin, "opencv_dullrazor"

    def _keras_mask_inpaint(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        input_shape = getattr(self.model, "input_shape", None)
        if not input_shape or len(input_shape) < 4:
            raise RuntimeError("Unsupported model input shape")

        target_h = int(input_shape[1] or 256)
        target_w = int(input_shape[2] or 256)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        pred = self.model.predict(x, verbose=0)
        pred = np.array(pred)

        if pred.ndim == 4:
            pred = pred[0]
        if pred.ndim == 3:
            pred = pred[..., 0]
        elif pred.ndim != 2:
            raise RuntimeError(f"Unexpected model output shape: {pred.shape}")

        pred = cv2.resize(pred.astype(np.float32), (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_bin = (pred > 0.5).astype(np.uint8) * 255
        if len(mask_bin.shape) == 3:
            mask_bin = mask_bin[:, :, 0]
        mask_bin = cv2.medianBlur(mask_bin.astype(np.uint8), 3)
        mask_inpaint = cv2.dilate(mask_bin, np.ones((3, 3), np.uint8), iterations=2)

        processed = cv2.inpaint(image_bgr, mask_inpaint, 3, cv2.INPAINT_TELEA)
        return processed, mask_bin, "keras_h5_mask_inpaint"

    def process(self, image_bytes: bytes) -> Dict:
        image_bgr = self._decode_image(image_bytes)

        if self.model is not None:
            try:
                processed, mask, method = self._keras_mask_inpaint(image_bgr)
            except Exception as exc:
                if not self.allow_fallback:
                    raise RuntimeError(f"Model inference failed and fallback disabled: {exc}")
                processed, mask, method = self._fallback_dullrazor(image_bgr)
                method = f"keras_runtime_fallback:{method}"
        else:
            if not self.allow_fallback:
                raise RuntimeError("Model is not loaded and fallback is disabled")
            processed, mask, method = self._fallback_dullrazor(image_bgr)
            method = f"no_model_fallback:{method}"

        return {
            "original_bgr": image_bgr,
            "processed_bgr": processed,
            "mask_gray": mask,
            "method": method,
            "model_backend": self.model_backend,
            "model_error": self.model_error,
            "mask_coverage_percent": round(float((mask > 0).sum()) * 100.0 / float(mask.size), 2),
        }

    def save_outputs(self, processed_bgr: np.ndarray, mask_gray: np.ndarray) -> Dict[str, str]:
        base = uuid.uuid4().hex
        processed_name = f"{base}_processed.jpg"
        mask_name = f"{base}_mask.png"

        processed_path = os.path.join(self.outputs_dir, processed_name)
        mask_path = os.path.join(self.outputs_dir, mask_name)

        ok_processed = cv2.imwrite(processed_path, processed_bgr)
        ok_mask = cv2.imwrite(mask_path, mask_gray)
        if not ok_processed or not ok_mask:
            raise RuntimeError("Failed to save output files")

        return {"processed_filename": processed_name, "mask_filename": mask_name}

