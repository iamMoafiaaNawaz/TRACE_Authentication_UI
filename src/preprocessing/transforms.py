# -*- coding: utf-8 -*-
"""
src/preprocessing/transforms.py
=================================
Image transforms for ConvNeXt-Base skin lesion classification.

Classes
-------
ResizePad
    Aspect-ratio-preserving resize + symmetric padding.
TransformBuilder
    Factory that builds training and evaluation transform pipelines.
"""

from typing import Tuple

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


class ResizePad:
    """
    Resize so the longest edge equals ``target_size``, then symmetrically
    pad the shorter edge to produce a square ``(target_size × target_size)``
    image — no squashing or stretching.

    Parameters
    ----------
    target_size : int
        Output square size in pixels.
    fill_color : tuple
        RGB fill for padding.  ``(0, 0, 0)`` (black) is standard for
        medical imaging; ``(127, 127, 127)`` is neutral-gray alternative.

    Example
    -------
    >>> tf = ResizePad(512)
    >>> out = tf(Image.open("lesion.jpg"))
    >>> out.size
    (512, 512)
    """

    def __init__(self, target_size: int, fill_color: Tuple[int, int, int] = (0, 0, 0)):
        self.target_size = target_size
        self.fill_color  = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        W, H  = img.size
        scale = self.target_size / max(W, H)
        new_W = int(round(W * scale))
        new_H = int(round(H * scale))
        img   = img.resize((new_W, new_H), Image.BILINEAR)

        pad_W      = self.target_size - new_W
        pad_H      = self.target_size - new_H
        pad_left   = pad_W // 2
        pad_right  = pad_W - pad_left
        pad_top    = pad_H // 2
        pad_bottom = pad_H - pad_top

        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom),
                      fill=self.fill_color)

    def __repr__(self) -> str:
        return f"ResizePad(target={self.target_size}, fill={self.fill_color})"


class TransformBuilder:
    """
    Builds training and evaluation transform pipelines for ConvNeXt.

    Uses ResizePad (aspect-ratio-safe), standard ImageNet normalisation,
    and dermatology-appropriate augmentations (flips, rotation, colour
    jitter, affine, blur, random erasing).

    Example
    -------
    >>> train_tf, eval_tf = TransformBuilder.build(image_size=512)
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    @classmethod
    def build(cls, image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Parameters
        ----------
        image_size : int
            Target square size for ResizePad.

        Returns
        -------
        (train_transform, eval_transform)
        """
        train_tf = transforms.Compose([
            ResizePad(image_size, fill_color=(0, 0, 0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10)),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(cls.IMAGENET_MEAN, cls.IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ])
        eval_tf = transforms.Compose([
            ResizePad(image_size, fill_color=(0, 0, 0)),
            transforms.ToTensor(),
            transforms.Normalize(cls.IMAGENET_MEAN, cls.IMAGENET_STD),
        ])
        return train_tf, eval_tf
