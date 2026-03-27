import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models as tv_models
from torchvision import transforms


class SkinClassifierService:
    """Loads ConvNeXt classifier once and exposes inference methods."""

    def __init__(
        self,
        model_path: str,
        classes: List[str],
        class_info: Dict[str, Dict],
        device: Optional[torch.device] = None,
    ):
        self.model_path = model_path
        self.classes = classes
        self.class_info = class_info
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[nn.Module] = None
        self.model_arch: Optional[str] = None
        self.model_error: Optional[str] = None

        self.transform = transforms.Compose(
            [
                # Standard ImageNet eval preprocessing for ConvNeXt-style classifiers.
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self) -> None:
        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            self.model = None
            self.model_arch = None
            self.model_error = f"Model not found: {self.model_path}"
            return

        try:
            m, arch, err = self._try_load_convnext_checkpoint(self.model_path, num_classes_hint=len(self.classes))
            if m is None:
                self.model = None
                self.model_arch = arch
                self.model_error = err or "Unknown error loading model"
                return

            self.model = m.to(self.device).eval()
            self.model_arch = arch
            self.model_error = None
        except Exception as exc:
            self.model = None
            self.model_arch = None
            self.model_error = str(exc)

    def predict_image_bytes(self, image_bytes: bytes) -> Dict:
        arr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Invalid image bytes")
        return self.predict_bgr(image_bgr)

    def predict_bgr(self, image_bgr: np.ndarray) -> Dict:
        if self.model is None:
            raise RuntimeError(f"Skin classifier is not loaded: {self.model_error}")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            probs = torch.softmax(out, dim=1).squeeze(0)

        if probs.ndim != 1 or probs.numel() != len(self.classes):
            raise ValueError(
                f"Unexpected classifier output shape: {tuple(probs.shape)} (expected {len(self.classes)} classes)"
            )

        probs_cpu = probs.detach().float().cpu().numpy().tolist()
        top_idx = int(np.argmax(probs_cpu))
        pred_code = self.classes[top_idx]
        confidence = float(probs_cpu[top_idx])
        raw_scores = {self.classes[i]: float(probs_cpu[i]) for i in range(len(self.classes))}

        info = self.class_info.get(pred_code, {})
        return {
            "class_code": pred_code,
            "diagnosis": info.get("name", pred_code),
            "result": info.get("type", pred_code),
            "severity": info.get("severity", "N/A"),
            "confidence": confidence,
            "raw_scores": raw_scores,
            "model": {"path": self.model_path, "arch": self.model_arch, "classes": self.classes},
        }

    @staticmethod
    def _strip_state_dict_prefixes(state_dict: Dict) -> Dict:
        fixed = {}
        for k, v in state_dict.items():
            if not isinstance(k, str):
                fixed[k] = v
                continue
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            fixed[nk] = v
        return fixed

    @staticmethod
    def _infer_num_classes_from_state_dict(state_dict: Dict) -> Optional[int]:
        candidates: List[int] = []
        for k, v in state_dict.items():
            if not isinstance(k, str) or not isinstance(v, torch.Tensor):
                continue
            if v.ndim != 2:
                continue
            kl = k.lower()
            if "classifier" not in kl and not kl.endswith("head.weight") and not kl.endswith("fc.weight"):
                continue
            out_features = int(v.shape[0])
            if 1 < out_features <= 100:
                candidates.append(out_features)
        if not candidates:
            return None
        return max(set(candidates), key=candidates.count)

    @staticmethod
    def _build_convnext(arch_name: str, num_classes: int) -> nn.Module:
        arch_name = (arch_name or "").lower().strip()
        constructors = {
            "convnext_tiny": getattr(tv_models, "convnext_tiny", None),
            "convnext_small": getattr(tv_models, "convnext_small", None),
            "convnext_base": getattr(tv_models, "convnext_base", None),
            "convnext_large": getattr(tv_models, "convnext_large", None),
        }

        ctor = constructors.get(arch_name)
        if ctor is None:
            raise ValueError(f"Unsupported ConvNeXt arch '{arch_name}' in this torchvision version")

        m = ctor(weights=None)
        if hasattr(m, "classifier") and isinstance(m.classifier, nn.Sequential):
            last_linear_idx = None
            for idx in range(len(m.classifier) - 1, -1, -1):
                if isinstance(m.classifier[idx], nn.Linear):
                    last_linear_idx = idx
                    break
            if last_linear_idx is None:
                raise ValueError("ConvNeXt classifier head not found (no nn.Linear in model.classifier)")
            in_features = int(m.classifier[last_linear_idx].in_features)
            m.classifier[last_linear_idx] = nn.Linear(in_features, int(num_classes))
        else:
            raise ValueError("Unexpected ConvNeXt model layout: missing nn.Sequential classifier")
        return m

    @staticmethod
    def _detect_flavor(state_dict: Dict) -> str:
        keys = [k for k in state_dict.keys() if isinstance(k, str)]
        # Heuristics for common ConvNeXt training stacks.
        if any(k.startswith("head.") or "head.fc" in k for k in keys):
            return "timm"
        if any(k.startswith("classifier.") for k in keys):
            return "torchvision"
        return "unknown"

    @staticmethod
    def _filter_to_model_keys(model: nn.Module, state_dict: Dict) -> Tuple[Dict, List[str]]:
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if isinstance(k, str) and k in model_keys}
        missing = [k for k in model_keys if k not in filtered]
        return filtered, missing

    def _try_load_convnext_checkpoint(
        self, path: str, num_classes_hint: int
    ) -> Tuple[Optional[nn.Module], Optional[str], Optional[str]]:
        ckpt = torch.load(path, map_location=self.device)

        if isinstance(ckpt, nn.Module):
            return ckpt, "custom_module", None

        if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), nn.Module):
            arch = str(ckpt.get("arch") or ckpt.get("model_name") or "checkpoint['model']")
            return ckpt["model"], arch, None

        if not isinstance(ckpt, dict):
            return None, None, "Invalid checkpoint format: expected nn.Module or dict checkpoint"

        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt.get("model")
        if not isinstance(state_dict, dict):
            return None, None, "Invalid checkpoint: missing state_dict/model_state_dict/model dict"

        state_dict = self._strip_state_dict_prefixes(state_dict)

        expected_classes = int(num_classes_hint)
        inferred_classes = self._infer_num_classes_from_state_dict(state_dict)
        if inferred_classes is not None and int(inferred_classes) != expected_classes:
            return None, None, (
                f"Checkpoint appears to have num_classes={inferred_classes} but service expects {expected_classes}. "
                "Fix the index-to-class mapping and num_classes to match training."
            )

        arch_hint = str(ckpt.get("arch") or ckpt.get("model_name") or ckpt.get("backbone") or "").lower().strip()
        if "convnext_tiny" in arch_hint:
            arch_hint = "convnext_tiny"
        elif "convnext_small" in arch_hint:
            arch_hint = "convnext_small"
        elif "convnext_base" in arch_hint:
            arch_hint = "convnext_base"
        elif "convnext_large" in arch_hint:
            arch_hint = "convnext_large"
        else:
            arch_hint = ""

        # Force the classifier head to match training: exactly N classes.
        num_classes = expected_classes

        flavor = self._detect_flavor(state_dict)
        if flavor == "timm":
            try:
                import timm  # type: ignore
            except Exception as exc:
                return None, None, f"Checkpoint looks like timm ConvNeXt (head.* keys) but timm is not available: {exc}"

            timm_arch = arch_hint or "convnext_base"
            try:
                m = timm.create_model(timm_arch, pretrained=False, num_classes=num_classes)  # type: ignore
            except Exception as exc:
                return None, timm_arch, f"Failed to create timm model '{timm_arch}': {exc}"

            # Require that classifier head weights are present and loaded.
            filtered, missing = self._filter_to_model_keys(m, state_dict)
            if any("head" in k for k in missing):
                return None, timm_arch, "Classifier head weights are missing; refusing partial load."
            if missing:
                return None, timm_arch, f"Missing keys for timm model load: {len(missing)}"

            m.load_state_dict(filtered, strict=True)
            return m, timm_arch, None

        candidate_arches = [a for a in [arch_hint] if a] + [
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large",
        ]
        last_arch: Optional[str] = None
        last_msg: Optional[str] = None
        for arch in candidate_arches:
            try:
                m = self._build_convnext(arch, num_classes).to(self.device)

                # Drop unrelated keys and require a complete match, including the classifier head.
                filtered, missing = self._filter_to_model_keys(m, state_dict)
                if missing:
                    # If the classifier head is missing, don't accept partial loads (causes "stuck" predictions).
                    if any("classifier" in k.lower() for k in missing):
                        last_arch = arch
                        last_msg = "Missing classifier head keys; refusing partial load."
                        continue
                    last_arch = arch
                    last_msg = f"Missing keys for torchvision load: {len(missing)}"
                    continue

                m.load_state_dict(filtered, strict=True)
                return m, arch, None
            except Exception as exc:
                last_arch = arch
                last_msg = str(exc)
                continue

        return None, last_arch, last_msg or "Failed to load ConvNeXt checkpoint"
