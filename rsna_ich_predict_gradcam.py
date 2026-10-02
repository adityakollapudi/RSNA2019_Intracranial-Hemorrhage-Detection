#!/usr/bin/env python3
"""
RSNA 2019 Intracranial Hemorrhage Detection - Unified Model Inference & Grad-CAM Pipeline.

Supports all model architectures from the 2DNet solution:
  - se_resnext101_32x4d (Image size: 256x256)
  - se_resnext50_32x4d  (Image size: 256x256)
  - DenseNet169_change_avg (Image size: 256x256)
  - DenseNet121_change_avg (Image size: 512x512)

Features:
  - Multi-backbone model loading from 2DNet/src/net/models.py
  - Automatic target layer selection for Grad-CAM / Layer-CAM per backbone
  - 3-slice context stacking (prev, current, next) or single-slice auto-expansion
  - DICOM windowing (brain/subdural window) and standard image support (PNG/JPG)
  - 80% center crop with coordinate re-projection to original CT resolution
  - Clean Python class API (ICHPredictor) for seamless project integration
  - Self-contained CLI with synthetic test mode (--test)
"""

import os
import sys
import json
import argparse
from typing import Union, List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Add 2DNet/src to sys.path so we can import from net.models if present
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "2DNet", "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import torchvision
except ImportError:
    torchvision = None

try:
    import pretrainedmodels
except ImportError:
    pretrainedmodels = None


def _get_densenet_features(arch_name, pretrained=False):
    """Safely obtain DenseNet feature extractor compatible across torchvision versions."""
    fn = getattr(torchvision.models, arch_name)
    try:
        if arch_name == 'densenet121':
            from torchvision.models import DenseNet121_Weights
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
        elif arch_name == 'densenet169':
            from torchvision.models import DenseNet169_Weights
            weights = DenseNet169_Weights.DEFAULT if pretrained else None
        else:
            weights = None
        return fn(weights=weights).features
    except (ImportError, AttributeError):
        return fn(pretrained=pretrained).features


class se_resnext50_32x4d(nn.Module):
    def __init__(self, pretrained=False):
        super(se_resnext50_32x4d, self).__init__()
        if pretrainedmodels is None:
            raise RuntimeError("pretrainedmodels is required. Install: pip install pretrainedmodels==0.7.4")
        self.model_ft = pretrainedmodels.__dict__['se_resnext50_32x4d'](
            num_classes=1000,
            pretrained='imagenet' if pretrained else None
        )
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))

    def forward(self, x):
        return self.model_ft(x)


class se_resnext101_32x4d(nn.Module):
    def __init__(self, pretrained=False):
        super(se_resnext101_32x4d, self).__init__()
        if pretrainedmodels is None:
            raise RuntimeError("pretrainedmodels is required. Install: pip install pretrainedmodels==0.7.4")
        self.model_ft = pretrainedmodels.__dict__['se_resnext101_32x4d'](
            num_classes=1000,
            pretrained='imagenet' if pretrained else None
        )
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))

    def forward(self, x):
        return self.model_ft(x)


class DenseNet169_change_avg(nn.Module):
    def __init__(self, pretrained=False):
        super(DenseNet169_change_avg, self).__init__()
        self.densenet169 = _get_densenet_features('densenet169', pretrained=pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1664, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet169(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


class DenseNet121_change_avg(nn.Module):
    def __init__(self, pretrained=False):
        super(DenseNet121_change_avg, self).__init__()
        self.densenet121 = _get_densenet_features('densenet121', pretrained=pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet121(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)
        return x

# The 6 intracranial hemorrhage labels predicted by all models
LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]

# Normalization parameters used across all 2DNet models
MEAN = np.array([0.456, 0.456, 0.456], dtype=np.float32)
STD = np.array([0.224, 0.224, 0.224], dtype=np.float32)

# Model Registry with default image sizes and target layers for Grad-CAM
MODEL_REGISTRY = {
    "se_resnext101_32x4d": {
        "class": se_resnext101_32x4d,
        "default_image_size": 256,
        "aliases": ["se_resnext101", "resnext101"],
        "get_layers": lambda m: [
            ("layer1", m.model_ft.layer1),
            ("layer2", m.model_ft.layer2),
            ("layer3", m.model_ft.layer3),
            ("layer4", m.model_ft.layer4),
        ],
    },
    "se_resnext50_32x4d": {
        "class": se_resnext50_32x4d,
        "default_image_size": 256,
        "aliases": ["se_resnext50", "resnext50"],
        "get_layers": lambda m: [
            ("layer1", m.model_ft.layer1),
            ("layer2", m.model_ft.layer2),
            ("layer3", m.model_ft.layer3),
            ("layer4", m.model_ft.layer4),
        ],
    },
    "DenseNet169_change_avg": {
        "class": DenseNet169_change_avg,
        "default_image_size": 256,
        "aliases": ["densenet169", "dense169", "densenet169_change_avg"],
        "get_layers": lambda m: [
            ("denseblock1", m.densenet169.denseblock1),
            ("denseblock2", m.densenet169.denseblock2),
            ("denseblock3", m.densenet169.denseblock3),
            ("denseblock4", m.densenet169.denseblock4),
        ],
    },
    "DenseNet121_change_avg": {
        "class": DenseNet121_change_avg,
        "default_image_size": 512,
        "aliases": ["densenet121", "dense121", "densenet121_change_avg"],
        "get_layers": lambda m: [
            ("denseblock1", m.densenet121.denseblock1),
            ("denseblock2", m.densenet121.denseblock2),
            ("denseblock3", m.densenet121.denseblock3),
            ("denseblock4", m.densenet121.denseblock4),
        ],
    },
}


def resolve_model_name(name: str) -> str:
    """Resolve aliases or case variations to canonical model names."""
    key = name.strip()
    if key in MODEL_REGISTRY:
        return key
    for canonical_name, config in MODEL_REGISTRY.items():
        if key.lower() == canonical_name.lower():
            return canonical_name
        if key.lower() in [a.lower() for a in config.get("aliases", [])]:
            return canonical_name
    valid = list(MODEL_REGISTRY.keys())
    raise ValueError(f"Unknown model name '{name}'. Choose from: {valid}")


def clean_state_dict(state: Any) -> Dict[str, torch.Tensor]:
    """Clean checkpoint state dict: strip DataParallel 'module.' prefix and unpack containers."""
    if isinstance(state, dict):
        for k in ["state_dict", "model_state_dict"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out


DEFAULT_CHECKPOINTS = [
    "model_epoch_best_4.pth",
    "model_epoch_best_0.pth",
    "model_epoch_best_1.pth",
    "model_epoch_best_2.pth",
    "model_epoch_best_3.pth",
    "seresnext101.pth",
    "resnext101_32x8d_wsl_checkpoint.pth",
]


def resolve_path(path: str) -> str:
    """Resolve file, checkpoint, or image path across current directory, script directory, checkpoints directories, parent directories, and user's Downloads folder."""
    if not path:
        return path
    clean = path.strip().strip("\"'").strip()
    if not clean:
        return clean

    # If directly exists as absolute or relative path
    if os.path.exists(clean):
        return os.path.abspath(clean)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    base_name = os.path.basename(clean)

    candidates = [
        os.path.join(cwd, clean),
        os.path.join(script_dir, clean),
        os.path.join(cwd, base_name),
        os.path.join(script_dir, base_name),
        os.path.join(cwd, "checkpoints", base_name),
        os.path.join(script_dir, "checkpoints", base_name),
        os.path.join(cwd, "..", clean),
        os.path.join(script_dir, "..", clean),
        os.path.join(script_dir, "..", "checkpoints", base_name),
        os.path.join(script_dir, "..", "..", clean),
        os.path.join(downloads, clean),
        os.path.join(downloads, base_name),
        os.path.join(downloads, "RSNA2019_Intracranial-Hemorrhage-Detection-master", base_name),
        os.path.join(downloads, "RSNA2019_Intracranial-Hemorrhage-Detection-master", "checkpoints", base_name),
        os.path.join(downloads, "rsna-master", base_name),
        os.path.join(downloads, "rsna-master", "rsna-master", base_name),
        os.path.join(downloads, "rsna-master", "rsna-master", "checkpoints", base_name),
        os.path.join(downloads, "rsna-master", "checkpoints", base_name),
    ]

    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    # Case-insensitive match in key folders
    for folder in [cwd, script_dir, os.path.join(cwd, "checkpoints"), os.path.join(script_dir, "checkpoints"), downloads]:
        if os.path.isdir(folder):
            try:
                for entry in os.listdir(folder):
                    if entry.lower() == base_name.lower():
                        return os.path.abspath(os.path.join(folder, entry))
            except Exception:
                pass

    return clean


def find_default_checkpoint() -> Optional[str]:
    """Auto-detect available trained checkpoint weights."""
    for name in DEFAULT_CHECKPOINTS:
        resolved = resolve_path(name)
        if os.path.exists(resolved):
            return resolved

    # Search in common directories for any .pth file matching model_epoch_best or seresnext
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    search_dirs = [
        cwd,
        os.path.join(cwd, "checkpoints"),
        script_dir,
        os.path.join(script_dir, "checkpoints"),
        os.path.join(script_dir, ".."),
        os.path.join(script_dir, "..", "checkpoints"),
        os.path.join(downloads, "RSNA2019_Intracranial-Hemorrhage-Detection-master"),
        os.path.join(downloads, "rsna-master", "rsna-master", "checkpoints"),
        os.path.join(downloads, "rsna-master", "rsna-master"),
        os.path.join(downloads, "rsna-master", "checkpoints"),
        os.path.join(downloads, "rsna-master"),
        downloads,
    ]

    for d in search_dirs:
        if os.path.isdir(d):
            try:
                for f in os.listdir(d):
                    if f.endswith(".pth") and ("model_epoch_best" in f or "resnext" in f.lower() or "ich" in f.lower()):
                        candidate = os.path.join(d, f)
                        if os.path.isfile(candidate):
                            return os.path.abspath(candidate)
            except Exception:
                pass
    return None


def load_model(
    model_name: str = "se_resnext101_32x4d",
    checkpoint: Optional[str] = "auto",
    device: Union[str, torch.device] = "cpu",
    pretrained: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Instantiate model and load checkpoint weights.
    If checkpoint is 'auto' or None, automatically searches for trained checkpoints like model_epoch_best_4.pth.
    """
    canonical_name = resolve_model_name(model_name)
    cfg = MODEL_REGISTRY[canonical_name]
    model_cls = cfg["class"]

    model = model_cls(pretrained=pretrained)

    has_checkpoint = False
    if checkpoint is None or (isinstance(checkpoint, str) and checkpoint.lower() in ["auto", "default"]):
        auto_ckpt = find_default_checkpoint()
        if auto_ckpt:
            checkpoint = auto_ckpt
            print(f"[Auto-Checkpoint] Found trained weights: {checkpoint}")
        else:
            checkpoint = None
    elif isinstance(checkpoint, str) and checkpoint.lower() in ["none", "untrained"]:
        checkpoint = None
    elif checkpoint:
        checkpoint = resolve_path(checkpoint)

    if checkpoint and os.path.exists(checkpoint):
        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint, map_location="cpu")
        state = clean_state_dict(ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [Notice] Missing keys ({len(missing)}): {missing[:6]}")
        if unexpected:
            print(f"  [Notice] Unexpected keys ({len(unexpected)}): {unexpected[:6]}")
        print(f"[Model] Trained weights successfully loaded from '{os.path.basename(checkpoint)}'")
        has_checkpoint = True
    elif checkpoint:
        print(f"[Warning] Checkpoint '{checkpoint}' not found! Running with initialized weights.")
    else:
        print(f"[Notice] No checkpoint specified. Model initialized with {'ImageNet' if pretrained else 'default'} weights.")

    device = torch.device(device)
    model.to(device).eval()
    model.has_checkpoint = has_checkpoint
    return model, cfg


def dicom_to_windowed(
    path: str,
    wc: Optional[float] = None,
    ww: Optional[float] = None,
) -> np.ndarray:
    """Read DICOM file, apply rescale slope/intercept and CT brain windowing."""
    try:
        import pydicom
    except ImportError:
        raise RuntimeError("pydicom is required to read DICOM files. Run: pip install pydicom")

    d = pydicom.dcmread(path)
    img = d.pixel_array.astype(np.float32)
    slope = float(getattr(d, "RescaleSlope", 1.0))
    intercept = float(getattr(d, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    # Default to DICOM embedded window or brain window (wc=40, ww=80)
    if wc is None:
        raw_wc = getattr(d, "WindowCenter", 40)
        if isinstance(raw_wc, pydicom.multival.MultiValue):
            raw_wc = raw_wc[0]
        wc = float(raw_wc)

    if ww is None:
        raw_ww = getattr(d, "WindowWidth", 80)
        if isinstance(raw_ww, pydicom.multival.MultiValue):
            raw_ww = raw_ww[0]
        ww = float(raw_ww)

    lo, hi = wc - ww / 2.0, wc + ww / 2.0
    img = np.clip(img, lo, hi)
    img = (img - lo) / max(hi - lo, 1e-6) * 255.0
    return img.astype(np.uint8)


def read_image(path: str) -> np.ndarray:
    """Read image from DICOM or standard image formats (PNG/JPEG/BMP)."""
    resolved = resolve_path(path)
    if resolved.lower().endswith(".dcm"):
        return dicom_to_windowed(resolved)
    img = cv2.imread(resolved, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return img


def create_synthetic_ct(size: int = 512) -> np.ndarray:
    """Generate a synthetic axial brain CT slice with simulated intracranial hemorrhage."""
    img = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    axes_outer = (int(size * 0.35), int(size * 0.43))
    axes_inner = (int(size * 0.32), int(size * 0.40))

    # Outer skull (high HU ~240)
    cv2.ellipse(img, center, axes_outer, 0, 0, 360, 240, int(size * 0.025))
    # Brain tissue (medium HU ~110)
    cv2.ellipse(img, center, axes_inner, 0, 0, 360, 110, -1)

    # Ventricles (CSF - lower intensity)
    v_left = (int(size * 0.45), int(size * 0.48))
    v_right = (int(size * 0.55), int(size * 0.48))
    cv2.ellipse(img, v_left, (int(size * 0.03), int(size * 0.07)), -15, 0, 360, 40, -1)
    cv2.ellipse(img, v_right, (int(size * 0.03), int(size * 0.07)), 15, 0, 360, 40, -1)

    # Simulated hyperdense hemorrhage spot (~195)
    bleed_pos = (int(size * 0.38), int(size * 0.40))
    cv2.circle(img, bleed_pos, int(size * 0.05), 195, -1)

    # Add subtle Gaussian noise for realism
    noise = np.random.normal(0, 3, (size, size)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def center_crop_80(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop 80% central region matching the competition training pipeline."""
    h, w = img.shape[:2]
    th, tw = int(h * 0.8), int(w * 0.8)
    y = (h - th) // 2
    x = (w - tw) // 2
    return img[y : y + th, x : x + tw], (x, y, tw, th)


def make_input(
    inputs: Union[str, List[str], np.ndarray],
    image_size: int = 256,
    apply_crop: bool = True,
) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int, int, int]]:
    """
    Prepare model input tensor:
      - Accepts 1 image path, 3 image paths [prev, curr, next], or preloaded numpy array
      - Stacks into 3-channel context
      - Performs 80% center crop
      - Resizes to image_size
      - Normalizes with RSNA mean & std
    """
    if isinstance(inputs, np.ndarray):
        if inputs.ndim == 2:
            imgs = [inputs, inputs, inputs]
        elif inputs.ndim == 3 and inputs.shape[2] == 3:
            imgs = [inputs[:, :, 0], inputs[:, :, 1], inputs[:, :, 2]]
        elif inputs.ndim == 3 and inputs.shape[2] == 1:
            g = inputs[:, :, 0]
            imgs = [g, g, g]
        else:
            raise ValueError(f"Unsupported numpy array shape: {inputs.shape}")
    elif isinstance(inputs, (list, tuple)):
        if len(inputs) == 1:
            im = read_image(inputs[0])
            imgs = [im, im, im]
        elif len(inputs) == 3:
            imgs = [read_image(p) for p in inputs]
        else:
            raise ValueError("Provide either 1 image path or exactly 3 consecutive slice paths.")
    elif isinstance(inputs, str):
        im = read_image(inputs)
        imgs = [im, im, im]
    else:
        raise TypeError(f"Invalid input type: {type(inputs)}")

    # Standardize dimensions to central slice
    h0, w0 = imgs[1].shape[:2]
    imgs = [cv2.resize(x, (w0, h0), interpolation=cv2.INTER_AREA) for x in imgs]
    stacked = np.stack(imgs, axis=-1).astype(np.uint8)

    original = stacked.copy()
    crop_box = (0, 0, w0, h0)
    if apply_crop:
        c, crop_box = center_crop_80(stacked)
    else:
        c = stacked

    c = cv2.resize(c, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = c.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, original, crop_box


class MultiLayerGradCAM:
    """Multi-layer Grad-CAM / Layer-CAM attribution engine."""

    def __init__(self, model: nn.Module, layers: List[Tuple[str, nn.Module]]):
        self.model = model
        self.layers = layers
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.handles = []
        for name, layer in layers:
            self.handles.append(layer.register_forward_hook(self._fwd(name)))
            self.handles.append(layer.register_full_backward_hook(self._bwd(name)))

    def _fwd(self, name: str):
        def hook(module, inp, out):
            self.activations[name] = out.detach()
        return hook

    def _bwd(self, name: str):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self.gradients[name] = grad_output[0].detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @staticmethod
    def cam_from_layer(act: torch.Tensor, grad: torch.Tensor, mode: str = "gradcam") -> torch.Tensor:
        # act / grad: [1, C, H, W] -> act[0], grad[0]: [C, H, W]
        act = act[0]
        grad = grad[0]
        if mode == "layercam":
            cam = (F.relu(grad) * F.relu(act)).sum(dim=0)
        else:
            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam = (weights * act).sum(dim=0)

        cam = F.relu(cam)
        cam = cam.unsqueeze(0).unsqueeze(0)
        return cam

    def generate(
        self,
        score: torch.Tensor,
        input_hw: Tuple[int, int],
        mode: str = "gradcam",
    ) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]], List[Dict[str, Any]]]:
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        maps, diagnostics = [], []
        for name, _ in self.layers:
            if name not in self.activations or name not in self.gradients:
                continue
            a, g = self.activations[name], self.gradients[name]
            cam = self.cam_from_layer(a, g, mode)
            cam = F.interpolate(cam, size=input_hw, mode="bilinear", align_corners=False)[0, 0]
            cam = cam.cpu().numpy()

            diff = cam.max() - cam.min()
            if diff > 1e-15:
                cam = (cam - cam.min()) / diff
            else:
                cam = np.zeros_like(cam)

            diagnostics.append({
                "layer": name,
                "activation_shape": list(a.shape),
                "gradient_shape": list(g.shape),
                "gradient_energy": float(g.abs().mean().cpu()),
                "activation_energy": float(a.abs().mean().cpu()),
                "cam_max": float(cam.max()),
            })
            maps.append((name, cam))

        if not maps:
            # Fallback to zero attribution map if gradient was completely clamped
            maps = [(self.layers[-1][0], np.zeros(input_hw, dtype=np.float32))]

        # Equal fusion across layers
        fused = np.mean([m for _, m in maps], axis=0)
        fused = np.maximum(fused, 0)
        f_diff = fused.max() - fused.min()
        if f_diff > 1e-15:
            fused = (fused - fused.min()) / f_diff
        else:
            fused = np.zeros_like(fused)

        return fused, maps, diagnostics


def overlay_heatmap(
    original_gray: np.ndarray,
    cam: np.ndarray,
    out_path: Optional[str] = None,
    alpha: float = 0.45,
    threshold: float = 0.25,
) -> np.ndarray:
    """Blend Grad-CAM jet colormap over grayscale CT slice with threshold suppression."""
    h, w = original_gray.shape[:2]
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam = np.clip(cam, 0, 1)

    mask = np.clip((cam - threshold) / max(1 - threshold, 1e-6), 0, 1)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    a = (mask * alpha)[..., None]
    out = (base * (1 - a) + heat * a).astype(np.uint8)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        cv2.imwrite(out_path, out)
    return out


def save_layer_grid(
    original: np.ndarray,
    layer_maps: List[Tuple[str, np.ndarray]],
    out_path: str,
    target_label: str,
) -> None:
    """Create a side-by-side grid of heatmaps from each layer."""
    thumbs = []
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(out_path)) or ".", "__tmp.png")
    for name, cam in layer_maps:
        ov = overlay_heatmap(original, cam, tmp_path, alpha=0.5, threshold=0.25)
        ov = cv2.resize(ov, (320, 320))
        cv2.putText(ov, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(ov)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    if not thumbs:
        return

    cols = min(3, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    canvas = np.zeros((rows * 320, cols * 320, 3), np.uint8)
    for i, im in enumerate(thumbs):
        canvas[(i // cols) * 320 : (i // cols + 1) * 320, (i % cols) * 320 : (i % cols + 1) * 320] = im

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, canvas)


class ICHPredictor:
    """
    High-level Python API for RSNA Intracranial Hemorrhage detection and Grad-CAM visualization.
    Designed for seamless integration into applications, pipelines, and web services.
    """

    def __init__(
        self,
        model_name: str = "se_resnext101_32x4d",
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        image_size: Optional[int] = None,
        pretrained: bool = False,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.canonical_name = resolve_model_name(model_name)
        self.model, self.cfg = load_model(
            self.canonical_name,
            checkpoint=checkpoint,
            device=self.device,
            pretrained=pretrained,
        )
        self.image_size = image_size if image_size is not None else self.cfg["default_image_size"]
        self.has_checkpoint = getattr(self.model, "has_checkpoint", False)

    def predict(
        self,
        inputs: Union[str, List[str], np.ndarray],
        target_class: str = "auto",
        mode: str = "gradcam",
        threshold: float = 0.25,
        apply_crop: bool = True,
        output_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run inference and generate Grad-CAM attribution.

        Parameters:
            inputs: DICOM path, PNG path, list of 3 slice paths, or numpy array.
            target_class: 'auto' (highest probability) or one of LABELS.
            mode: 'gradcam' or 'layercam'.
            threshold: Heatmap threshold [0.0 - 1.0].
            apply_crop: Whether to apply 80% center crop.
            output_prefix: If specified, saves dashboard, heatmap, overlay, and JSON files.

        Returns:
            Dictionary containing predicted probabilities, target class, raw CAM arrays,
            and file paths (if output_prefix is set).
        """
        x, original_rgb, crop_box = make_input(inputs, self.image_size, apply_crop=apply_crop)
        x = x.to(self.device)

        layers = self.cfg["get_layers"](self.model)
        cam_engine = MultiLayerGradCAM(self.model, layers)

        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0]

            if target_class == "auto":
                target_idx = int(torch.argmax(probs).item())
            elif target_class in LABELS:
                target_idx = LABELS.index(target_class)
            else:
                raise ValueError(f"Unknown target class '{target_class}'. Choose from {LABELS} or 'auto'.")

            score = logits[0, target_idx]
            fused, layer_maps, diagnostics = cam_engine.generate(
                score, (self.image_size, self.image_size), mode=mode
            )
        cam_engine.remove()

        prob_dict = {lab: float(p) for lab, p in zip(LABELS, probs.detach().cpu().numpy())}
        target_name = LABELS[target_idx]
        target_score = float(score.detach().cpu().item())

        # Central slice from the 3-channel input
        central_slice = original_rgb[:, :, 1]

        # Re-project CAM back to uncropped original image dimensions
        crop_x, crop_y, crop_w, crop_h = crop_box
        cam_crop = cv2.resize(fused, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        cam_full = np.zeros(central_slice.shape, dtype=np.float32)
        cam_full[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w] = cam_crop
        if cam_full.max() > 1e-15:
            cam_full = cam_full / cam_full.max()

        overlay_img = overlay_heatmap(central_slice, cam_full, alpha=0.5, threshold=threshold)

        result: Dict[str, Any] = {
            "model": self.canonical_name,
            "target_class": target_name,
            "target_score": target_score,
            "probabilities": prob_dict,
            "crop_box_xywh": list(map(int, crop_box)),
            "fused_cam": cam_full,
            "overlay_bgr": overlay_img,
            "central_slice": central_slice,
            "layers": diagnostics,
            "outputs": {},
        }

        if output_prefix:
            stem = output_prefix
            if stem.endswith(".png") or stem.endswith(".jpg"):
                stem = os.path.splitext(stem)[0]

            dashboard_path = stem + "_dashboard.png"
            cam_path = stem + "_heatmap.png"
            grid_path = stem + "_layers.png"
            overlay_path = stem + "_overlay.png"
            json_path = stem + ".json"

            cv2.imwrite(cam_path, (cam_full * 255).astype(np.uint8))
            cv2.imwrite(overlay_path, overlay_img)
            save_layer_grid(
                central_slice,
                [(n, cv2.resize(m, (central_slice.shape[1], central_slice.shape[0]))) for n, m in layer_maps],
                grid_path,
                target_name,
            )

            # Build side-by-side dashboard
            panels = [
                cv2.resize(cv2.cvtColor(central_slice, cv2.COLOR_GRAY2BGR), (512, 512)),
                cv2.resize(overlay_img, (512, 512)),
            ]
            dashboard = np.hstack(panels)
            cv2.putText(
                dashboard,
                "Original CT Central Slice",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                dashboard,
                f"{self.canonical_name} CAM: {target_name} ({prob_dict[target_name]:.3f})",
                (530, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            cv2.imwrite(dashboard_path, dashboard)

            json_summary = {
                "model": self.canonical_name,
                "target_class": target_name,
                "target_score": target_score,
                "probabilities": prob_dict,
                "mode": mode,
                "crop_box_xywh": list(map(int, crop_box)),
                "layers": diagnostics,
                "outputs": {
                    "dashboard": dashboard_path,
                    "heatmap": cam_path,
                    "overlay": overlay_path,
                    "layer_grid": grid_path,
                },
            }
            with open(json_path, "w") as f:
                json.dump(json_summary, f, indent=2)

            result["outputs"] = json_summary["outputs"]
            result["outputs"]["json"] = json_path

        return result


def show_popup_window(
    original_img: np.ndarray,
    overlay_bgr: np.ndarray,
    model_name: str,
    target_class: str,
    target_score: float,
    probabilities: Dict[str, float],
    image_label: str = "Input CT Slice",
    safe_threshold: float = 0.50,
    has_checkpoint: bool = True,
) -> None:
    """
    Display a GUI popup window:
      - If safe (no hemorrhage detected): Styled in GREEN
      - If unsafe (hemorrhage detected): Styled in RED
      - If untrained weights: Styled in AMBER with demo warning
      - Top: Status banner & model architecture
      - Left: Original CT image with status border
      - Right: Grad-CAM attribution overlay with status badge
      - Bottom: Model prediction & confidence breakdown
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Determine whether the brain is safe
    any_prob = probabilities.get("any", 0.0)
    subtype_probs = [v for k, v in probabilities.items() if k != "any"]
    max_subtype_prob = max(subtype_probs) if subtype_probs else any_prob
    overall_risk = max(any_prob, max_subtype_prob)

    is_safe = overall_risk < safe_threshold

    # Define color scheme: Amber for untrained demo, Green for safe, Red for hemorrhage
    if not has_checkpoint:
        theme_color = "#ffb703"   # Amber warning
        theme_bg = "#3d2b00"      # Dark amber background
        status_title = "DEMO MODE  -  UNTRAINED WEIGHTS (RANDOM ~50%)"
        status_subtitle = "Weights are not loaded! Download and pass a .pth checkpoint for real medical diagnosis."
        badge_label = "DEMO: UNTRAINED WEIGHTS"
        orig_label = "Original CT Slice\n[Untrained Baseline]"
        cam_label = "Grad-CAM Attribution Overlay\n[Random Baseline Activation]"
    elif is_safe:
        theme_color = "#00e676"   # Vibrant medical green
        theme_bg = "#073b1a"      # Dark green badge background
        status_title = "BRAIN IS SAFE  -  NO HEMORRHAGE DETECTED"
        status_subtitle = f"All hemorrhage markers are within normal limits (Confidence: {(1.0 - overall_risk) * 100:.1f}% Normal)"
        badge_label = "STATUS: SAFE (NORMAL)"
        orig_label = "Original CT Slice\n[Normal Brain Parenchyma]"
        cam_label = "Grad-CAM Attribution Overlay\n[Status: Normal / Safe]"
    else:
        theme_color = "#ff1744"   # Vivid alert red
        theme_bg = "#3d080e"      # Dark crimson badge background
        status_title = f"ALERT: INTRACRANIAL HEMORRHAGE DETECTED  -  {target_class.upper()}"
        status_subtitle = f"Elevated hemorrhage probability detected (Risk: {overall_risk * 100:.1f}%)"
        badge_label = f"ALERT: {target_class.upper()} DETECTED"
        orig_label = "Original CT Slice\n[Abnormal CT Scan]"
        cam_label = f"Grad-CAM Attribution Overlay\n[HEMORRHAGE SITE: {target_class.upper()} ({probabilities.get(target_class, overall_risk) * 100:.1f}%)]"

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(14, 9.2), facecolor="#18181f")
    try:
        fig.canvas.manager.set_window_title(f"RSNA ICH Detection | {'SAFE (Green)' if is_safe else 'ALERT (Red)'} | {model_name}")
    except Exception:
        pass

    gs = gridspec.GridSpec(2, 2, height_ratios=[2.2, 1.2], hspace=0.34, wspace=0.20)

    # 1. Left Side: Original Image
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original_img, cmap="gray")
    ax_orig.set_title(
        f"{orig_label}\nScan: {image_label}",
        fontsize=12,
        fontweight="bold",
        pad=8,
        color=theme_color,
    )
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])
    for spine in ax_orig.spines.values():
        spine.set_color(theme_color)
        spine.set_linewidth(3.0)
        spine.set_visible(True)

    # 2. Right Side: Grad-CAM Overlay
    ax_cam = fig.add_subplot(gs[0, 1])
    ax_cam.imshow(overlay_rgb)
    ax_cam.set_title(
        cam_label,
        fontsize=12,
        fontweight="bold",
        pad=8,
        color=theme_color,
    )
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])
    for spine in ax_cam.spines.values():
        spine.set_color(theme_color)
        spine.set_linewidth(3.0)
        spine.set_visible(True)

    # Status badge on upper corner of CAM image
    ax_cam.text(
        0.03,
        0.94,
        badge_label,
        transform=ax_cam.transAxes,
        color="#ffffff",
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=theme_color,
            edgecolor="#ffffff",
            linewidth=1.5,
            alpha=0.95,
        ),
    )

    # 3. Bottom Panel: Model Prediction & Confidence Breakdown
    ax_pred = fig.add_subplot(gs[1, :])
    ax_pred.set_facecolor("#22222c")

    classes = list(probabilities.keys())
    scores = [probabilities[c] * 100 for c in classes]

    # Invert so top class displays at the top of the horizontal bar chart
    classes_rev = classes[::-1]
    scores_rev = scores[::-1]

    # Assign bar colors based on status
    bar_colors = []
    for c, s in zip(classes_rev, scores_rev):
        if not has_checkpoint:
            # Amber neutral tone for demo mode
            bar_colors.append("#ffb703" if s == max(scores_rev) else "#e0a000")
        elif is_safe:
            # All vibrant green when safe
            bar_colors.append("#00e676" if s == max(scores_rev) else "#00c853")
        else:
            # All red when hemorrhage is detected
            bar_colors.append("#ff1744" if s >= 50 else ("#ff5252" if s >= 20 else "#c62828"))

    bars = ax_pred.barh(classes_rev, scores_rev, color=bar_colors, height=0.62, edgecolor="#ffffff", linewidth=0.6)
    ax_pred.set_xlim(0, 108)
    ax_pred.set_xlabel("Predicted Probability (%)", fontsize=11, fontweight="bold", color="#ffffff")
    ax_pred.tick_params(colors="#ffffff", labelsize=10)
    for spine in ax_pred.spines.values():
        spine.set_color("#444455")
    ax_pred.spines["left"].set_color(theme_color)
    ax_pred.spines["left"].set_linewidth(2.5)

    # Header banner above the bar chart
    if not has_checkpoint:
        bottom_title = "[ DEMO MODE ]  Untrained Weights (Random ~50% Baseline)  |  No Diagnosis Available"
    elif is_safe:
        bottom_title = f"[ SAFE ]  Normal Scan  |  No Hemorrhage Detected  |  Safe Score: {(1.0 - overall_risk) * 100:.1f}%"
    else:
        bottom_title = f"[ ALERT ]  Hemorrhage Detected  |  Primary Finding: {target_class.upper()}  |  Confidence: {overall_risk * 100:.1f}%"

    ax_pred.set_title(
        bottom_title,
        fontsize=12,
        fontweight="bold",
        pad=10,
        color=theme_color,
    )
    ax_pred.grid(axis="x", linestyle="--", alpha=0.25, color="#888899")

    # Numeric percentage badges
    for bar, score in zip(bars, scores_rev):
        w = bar.get_width()
        ax_pred.text(
            w + 1.2,
            bar.get_y() + bar.get_height() / 2.0,
            f"{score:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#ffffff",
        )

    # Top Super Title with status banner
    plt.suptitle(
        f"{status_title}\n{status_subtitle}  |  Model: {model_name}",
        fontsize=14,
        fontweight="bold",
        color=theme_color,
        y=0.98,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=theme_bg,
            edgecolor=theme_color,
            linewidth=2.0,
        ),
    )

    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.95, hspace=0.36, wspace=0.20)
    print(f"\n[Display] Popup window opened ({'SAFE: Green' if is_safe else 'ALERT: Red'}). Close window when finished.")
    plt.show()


def format_ascii_bar(pct: float, width: int = 20) -> str:
    """Generate a clean visual ASCII bar for probability percentage compatible with all Windows consoles."""
    filled = int(round((pct / 100.0) * width))
    filled = max(0, min(width, filled))
    return "#" * filled + "-" * (width - filled)


def clean_user_input_path(raw: str) -> str:
    """Clean path string from terminal input, stripping quotes, powershell '&' prefix, and extra spaces."""
    s = raw.strip()
    if s.startswith("&"):
        s = s[1:].strip()
    return s.strip("\"'").strip()


def _execute_and_display(
    predictor: ICHPredictor,
    input_data: Any,
    image_label: str,
    args: Any,
) -> None:
    """Run prediction, print formatted clinical diagnosis in terminal with colors, and display popup window."""
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    import re
    base_clean = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', os.path.splitext(image_label)[0])
    default_output_prefix = os.path.join(output_dir, f"{base_clean}_gradcam")

    output_prefix = os.path.splitext(args.output)[0] if args.output else default_output_prefix

    result = predictor.predict(
        inputs=input_data,
        target_class=args.target,
        mode=args.mode,
        threshold=args.threshold,
        apply_crop=not args.no_crop,
        output_prefix=output_prefix,
    )

    has_checkpoint = getattr(predictor, "has_checkpoint", False)

    any_p = result["probabilities"].get("any", 0.0)
    subtype_p = [v for k, v in result["probabilities"].items() if k != "any"]
    max_subtype_p = max(subtype_p) if subtype_p else any_p
    overall_p = max(any_p, max_subtype_p)
    is_safe = overall_p < args.safe_threshold

    # ANSI Colors for terminal
    COLOR_RESET = "\033[0m"
    COLOR_GREEN = "\033[92m"
    COLOR_RED = "\033[91m"
    COLOR_YELLOW = "\033[93m"
    COLOR_BOLD = "\033[1m"

    if not has_checkpoint:
        term_color = COLOR_YELLOW
        diag_title = "[ DEMO / UNTRAINED ] (Random baseline ~50% - load a .pth file)"
    elif is_safe:
        term_color = COLOR_GREEN
        diag_title = f"[ SAFE ] - Brain is Safe / No Hemorrhage Detected  |  Normal Confidence: {(1.0 - overall_p) * 100:.2f}%"
    else:
        term_color = COLOR_RED
        diag_title = f"[ ALERT ] - Hemorrhage Detected: {result['target_class'].upper()}  |  Risk Level: {overall_p * 100:.2f}%"

    print("\n" + term_color + "=" * 68)
    print(f"  SCAN: {image_label}")
    print(f"  MODEL ARCHITECTURE: {result['model']}")
    print(f"  DIAGNOSIS: {diag_title}")
    print("=" * 68 + COLOR_RESET)

    print(f"  {'CLASS':<20s} {'PROBABILITY':<12s} {'VISUAL BAR (0-100%)':<24s}")
    print("-" * 68)

    for lab, p in result["probabilities"].items():
        pct = p * 100.0
        bar = format_ascii_bar(pct, width=20)
        if not has_checkpoint:
            status_txt = "Demo"
            c_tag = COLOR_YELLOW
        elif is_safe:
            status_txt = "Safe / Normal"
            c_tag = COLOR_GREEN
        else:
            if pct >= 50:
                status_txt = "HIGH RISK ALERT"
                c_tag = COLOR_RED
            elif pct >= 25:
                status_txt = "Warning"
                c_tag = COLOR_YELLOW
            else:
                status_txt = "Normal"
                c_tag = COLOR_GREEN

        print(f"  {c_tag}{lab:<20s} : {pct:6.2f}%   [{bar}] {status_txt}{COLOR_RESET}")

    print("-" * 68)
    print("  [Grad-CAM] Extracted gradients across layers (layer1, layer2, layer3, layer4) & projected onto head CT")
    if result.get("outputs"):
        for k, v in result["outputs"].items():
            print(f"  Saved {k.capitalize()}: {v}")
    print("=" * 68)

    # Show popup window by default unless user passed --no-popup
    if not getattr(args, "no_popup", False):
        print(f"\n[Display] Opening interactive diagnostic popup window ({'SAFE: Green' if is_safe else 'ALERT: Red'})...")
        show_popup_window(
            original_img=result["central_slice"],
            overlay_bgr=result["overlay_bgr"],
            model_name=result["model"],
            target_class=result["target_class"],
            target_score=result["target_score"],
            probabilities=result["probabilities"],
            image_label=image_label,
            safe_threshold=args.safe_threshold,
            has_checkpoint=has_checkpoint,
        )


def main():
    ap = argparse.ArgumentParser(
        description="RSNA 2019 Intracranial Hemorrhage Detection - Terminal Prediction & Grad-CAM Pipeline."
    )
    ap.add_argument(
        "--model",
        "--backbone",
        dest="model",
        default="se_resnext101_32x4d",
        help=f"Model architecture. Choices: {list(MODEL_REGISTRY.keys())} or aliases (e.g. densenet121, resnext101).",
    )
    ap.add_argument(
        "--checkpoint",
        default="auto",
        help="Path to trained PyTorch checkpoint (.pth). Defaults to auto-detecting model_epoch_best_4.pth.",
    )
    ap.add_argument(
        "--image",
        default=None,
        help="Path to DICOM/PNG/JPG image, or comma-separated list of 3 consecutive slices (prev, curr, next).",
    )
    ap.add_argument(
        "--test",
        action="store_true",
        help="Run self-contained test using synthetic brain CT slice.",
    )
    ap.add_argument(
        "--safe-threshold",
        type=float,
        default=0.50,
        help="Probability threshold below which the brain is considered safe/normal (default: 0.50).",
    )
    ap.add_argument(
        "--no-popup",
        action="store_true",
        help="Disable GUI popup window (by default, popup window opens automatically).",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Custom output path prefix if saving to disk.",
    )
    ap.add_argument("--mode", choices=["gradcam", "layercam"], default="gradcam")
    ap.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input resolution (default: 512 for DenseNet121, 256 for other backbones).",
    )
    ap.add_argument("--no-crop", action="store_true", help="Disable 80 percent center crop.")
    ap.add_argument("--target", default="auto", choices=["auto"] + LABELS, help="Target class for Grad-CAM.")
    ap.add_argument("--threshold", type=float, default=0.25, help="Heatmap suppression threshold.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    # Determine device
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    print("\n" + "=" * 68)
    print("  RSNA 2019 Intracranial Hemorrhage Detection & Grad-CAM Pipeline")
    print("=" * 68)
    print(f"Device: {device_str}")

    # Initialize Predictor (loads model once)
    predictor = ICHPredictor(
        model_name=args.model,
        checkpoint=args.checkpoint,
        device=device_str,
        image_size=args.image_size,
    )
    print(f"Active Backbone: {predictor.canonical_name} (Resolution: {predictor.image_size}x{predictor.image_size})")

    # If --image or --test provided directly on CLI, run once and exit
    if args.image is not None or args.test:
        if args.test or args.image is None:
            print("Generating synthetic brain CT slice for verification...")
            input_data = create_synthetic_ct(size=512)
            label = "Demo Synthetic Brain CT"
        else:
            raw_paths = [clean_user_input_path(x) for x in args.image.split(",")]
            input_paths = [resolve_path(p) for p in raw_paths]
            missing = [p for p in input_paths if not os.path.exists(p)]
            if missing:
                print(f"[Error] Image file not found: '{missing[0]}'")
                return
            input_data = input_paths
            label = os.path.basename(input_paths[len(input_paths) // 2])
        _execute_and_display(predictor, input_data, label, args)
        return

    # Interactive Terminal Loop
    print("\nInstructions:")
    print("  - Paste or type the image file address/name (DICOM / PNG / JPG)")
    print("  - Press Enter with empty input to run demo synthetic CT")
    print("  - Type 'q' or 'exit' to quit\n")

    while True:
        try:
            user_input = input("Enter CT image address: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        cleaned = clean_user_input_path(user_input)

        if cleaned.lower() in ["q", "quit", "exit"]:
            print("Exiting.")
            break

        if not cleaned:
            print("\n[Demo] Generating synthetic brain CT scan...")
            img_data = create_synthetic_ct(size=512)
            _execute_and_display(predictor, img_data, "Demo Synthetic Brain CT", args)
            continue

        raw_paths = [clean_user_input_path(p) for p in cleaned.split(",")]
        paths = [resolve_path(p) for p in raw_paths]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"\n[Error] File not found: '{missing[0]}'")
            print("Tips: Check spelling, paste the full path, or place the file in Downloads / current folder.\n")
            continue

        label = os.path.basename(paths[len(paths) // 2])
        _execute_and_display(predictor, paths, label, args)


if __name__ == "__main__":
    main()

