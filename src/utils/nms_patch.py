# -*- coding: utf-8 -*-
"""
src/utils/nms_patch.py
=======================
Patches the ultralytics NMS function for DDP compatibility.

Two strategies applied together
---------------------------------
1. **Disk patch** (:func:`patch_ultralytics_nms_on_disk`)
   Rewrites ``ultralytics/utils/nms.py`` on disk so DDP worker subprocesses
   inherit the fix.  Must be called before any ``model.train()``.

2. **Runtime patch** (:func:`apply_runtime_nms_patch`)
   Monkey-patches ``torchvision.ops.nms`` in the current process so that
   any NMS call silently falls back to CPU if the GPU raises
   ``NotImplementedError`` or ``RuntimeError``.

Why this is needed
------------------
Some GPU configurations (e.g. multi-GPU DDP on nodes without full
``torchvision`` CUDA support) raise ``NotImplementedError`` inside
``torchvision.ops.nms``.  The disk patch ensures DDP worker subprocesses
(which re-import ultralytics fresh) also have the fix.
"""

import shutil
import traceback
from pathlib import Path


# ==============================================================================
# DISK PATCH
# ==============================================================================

def patch_ultralytics_nms_on_disk() -> bool:
    """
    Rewrite ``ultralytics/utils/nms.py`` to add a CPU fallback around
    the torchvision NMS call.

    Returns ``True`` on success, ``False`` if the patch could not be applied
    (e.g. ultralytics not installed, target line not found).

    A ``.orig`` backup is created on first run.
    """
    try:
        import ultralytics
        nms_path = Path(ultralytics.__file__).parent / "utils" / "nms.py"
        if not nms_path.exists():
            print(f"[nms_patch] nms.py not found: {nms_path}")
            return False

        src = nms_path.read_text(encoding="utf-8")
        if "CPU_NMS_FALLBACK" in src:
            print("[nms_patch] Already patched.")
            return True

        TARGET = "i = torchvision.ops.nms(boxes, scores, iou_thres)"
        if TARGET not in src:
            print("[nms_patch] Target line not found — add --single_gpu if DDP crashes")
            return False

        REPLACEMENT = (
            "# CPU_NMS_FALLBACK\n"
            "            try:\n"
            "                i = torchvision.ops.nms(boxes, scores, iou_thres)\n"
            "            except (NotImplementedError, RuntimeError):\n"
            "                i = torchvision.ops.nms(\n"
            "                    boxes.cpu().float(),\n"
            "                    scores.cpu().float(),\n"
            "                    iou_thres).to(boxes.device)"
        )

        # Backup original
        bak = nms_path.with_suffix(".py.orig")
        if not bak.exists():
            shutil.copy2(nms_path, bak)

        patched = src.replace("            " + TARGET, REPLACEMENT, 1)
        if patched == src:
            patched = src.replace(TARGET, REPLACEMENT.strip(), 1)

        nms_path.write_text(patched, encoding="utf-8")
        print(f"[nms_patch] SUCCESS — patched {nms_path}")
        return True

    except Exception as e:
        print(f"[nms_patch] ERROR: {e}")
        traceback.print_exc()
        return False


# ==============================================================================
# RUNTIME PATCH
# ==============================================================================

def apply_runtime_nms_patch() -> None:
    """
    Monkey-patch ``torchvision.ops.nms`` in the current process to add a
    silent CPU fallback.

    Safe to call multiple times — subsequent calls are no-ops if the patch
    is already applied.
    """
    try:
        import torchvision.ops as tv
        import torchvision.ops.boxes as tvb

        if getattr(tv.nms, "_cpu_fallback_patched", False):
            return  # already patched

        _orig = tv.nms

        def _safe(b, s, t):
            try:
                return _orig(b, s, t)
            except (NotImplementedError, RuntimeError):
                return _orig(
                    b.cpu().float(),
                    s.cpu().float(),
                    t,
                ).to(b.device)

        _safe._cpu_fallback_patched = True
        tv.nms  = _safe
        tvb.nms = _safe

    except Exception:
        pass