from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121

from utils import VERTEBRA_LABELS, extract_centered_label_cube


CLASS_NAMES_4 = ["none", "blastic", "lytic", "mixed"]
CLASS_NAMES_2 = ["none", "cancer"]


@dataclass
class CAMResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_class: int
    target_class: int
    cam: np.ndarray


def infer_num_classes_from_checkpoint(model_path, default=4):
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    for key, value in state_dict.items():
        if key.endswith("class_layers.out.weight") or key.endswith("class_layers.3.weight"):
            return int(value.shape[0])
    return default


def build_stage1_model(num_classes=4):
    return DenseNet121(spatial_dims=3, in_channels=1, out_channels=int(num_classes))


def load_stage1_model(model_path, device=None, num_classes=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = str(model_path)
    if num_classes is None:
        num_classes = infer_num_classes_from_checkpoint(model_path)
    model = build_stage1_model(num_classes=num_classes)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def class_names_for_count(num_classes):
    if int(num_classes) == 2:
        return CLASS_NAMES_2
    if int(num_classes) == 4:
        return CLASS_NAMES_4
    return [str(i) for i in range(int(num_classes))]


def normalize_stage1_patch(ct_patch, mask, norm_mode="zscore_sigmoid", zscore_scale=1.5, foreground_floor=0.15):
    out = np.zeros_like(ct_patch, dtype=np.float32)
    if not np.any(mask):
        return out

    vals = ct_patch[mask]
    if norm_mode == "robust_minmax":
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if hi <= lo:
            lo = float(vals.min())
            hi = float(vals.max())
        if hi > lo:
            out = (ct_patch - lo) / (hi - lo)
            out = np.clip(out, 0.0, 1.0).astype(np.float32)
    elif norm_mode == "zscore_sigmoid":
        mu = float(np.mean(vals))
        sigma = max(float(np.std(vals)), 1e-6)
        z = (ct_patch - mu) / (sigma * max(float(zscore_scale), 1e-6))
        z = np.clip(z, -12.0, 12.0)
        out = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    elif norm_mode == "ct_hu_window":
        out = ((ct_patch + 200.0) / 1200.0).astype(np.float32)
        out = np.clip(out, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    floor = float(np.clip(foreground_floor, 0.0, 0.95))
    out = floor + out * (1.0 - floor)
    out[~mask] = 0.0
    return out.astype(np.float32)


def build_stage1_patch_and_mask(
    ct,
    seg,
    vertebra,
    patch_size=(96, 96, 64),
    norm_mode="zscore_sigmoid",
    zscore_scale=1.5,
    foreground_floor=0.15,
):
    vname = str(vertebra).upper()
    if vname not in VERTEBRA_LABELS:
        raise ValueError(f"Unknown vertebra: {vertebra}")

    label = VERTEBRA_LABELS[vname]
    seg_bin = np.where(seg == label, 1.0, 0.0)
    ct_patch = extract_centered_label_cube(ct, seg, label, size=tuple(patch_size))[0].astype(np.float32)
    mask = extract_centered_label_cube(seg_bin, seg, label, size=tuple(patch_size))[0] > 0.5
    if norm_mode != "robust_minmax":
        ct_patch = np.clip(ct_patch, -200.0, 1000.0)
    patch = normalize_stage1_patch(
        ct_patch,
        mask,
        norm_mode=norm_mode,
        zscore_scale=zscore_scale,
        foreground_floor=foreground_floor,
    )
    return torch.from_numpy(patch[None, ...]).float(), mask


def build_stage1_patch_mask_and_coords(
    ct,
    seg,
    vertebra,
    patch_size=(96, 96, 64),
    norm_mode="zscore_sigmoid",
    zscore_scale=1.5,
    foreground_floor=0.15,
):
    vname = str(vertebra).upper()
    if vname not in VERTEBRA_LABELS:
        raise ValueError(f"Unknown vertebra: {vertebra}")

    label = VERTEBRA_LABELS[vname]
    if ct.shape != seg.shape:
        common_shape = tuple(min(v, m) for v, m in zip(ct.shape, seg.shape))
        ct = ct[:common_shape[0], :common_shape[1], :common_shape[2]]
        seg = seg[:common_shape[0], :common_shape[1], :common_shape[2]]

    target_mask = seg == label
    if not np.any(target_mask):
        size = tuple(int(v) for v in patch_size)
        patch = torch.zeros((1, *size), dtype=torch.float32)
        return patch, np.zeros(size, dtype=bool), np.zeros((*size, 3), dtype=np.float32)

    coords = np.argwhere(target_mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    src_ct = np.where(target_mask, ct, 0.0)[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]].astype(np.float32)
    src_mask = target_mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    src_shape_original = np.asarray(src_ct.shape, dtype=np.float32)

    if any(src_ct.shape[d] > patch_size[d] for d in range(3)):
        scale = min(patch_size[d] / src_ct.shape[d] for d in range(3))
        new_shape = tuple(max(1, int(round(src_ct.shape[d] * scale))) for d in range(3))
        src_ct_t = torch.from_numpy(src_ct).float()[None, None]
        src_ct = F.interpolate(src_ct_t, size=new_shape, mode="trilinear", align_corners=False)[0, 0].cpu().numpy()
        src_mask_t = torch.from_numpy(src_mask.astype(np.float32))[None, None]
        src_mask = F.interpolate(src_mask_t, size=new_shape, mode="nearest")[0, 0].cpu().numpy() > 0.5

    size = tuple(int(v) for v in patch_size)
    ct_patch = np.zeros(size, dtype=np.float32)
    mask_patch = np.zeros(size, dtype=bool)
    coord_patch = np.zeros((*size, 3), dtype=np.float32)

    src_slices = []
    dst_slices = []
    for dim, dim_size in enumerate(size):
        src_dim = src_ct.shape[dim]
        if src_dim <= dim_size:
            dst_start = (dim_size - src_dim) // 2
            dst_end = dst_start + src_dim
            src_start = 0
            src_end = src_dim
        else:
            src_start = (src_dim - dim_size) // 2
            src_end = src_start + dim_size
            dst_start = 0
            dst_end = dim_size
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    ct_patch[dst_slices[0], dst_slices[1], dst_slices[2]] = src_ct[src_slices[0], src_slices[1], src_slices[2]]
    mask_patch[dst_slices[0], dst_slices[1], dst_slices[2]] = src_mask[src_slices[0], src_slices[1], src_slices[2]]

    src_used_shape = np.asarray(src_ct.shape, dtype=np.float32)
    scale_to_original = src_shape_original / np.maximum(src_used_shape, 1.0)
    local_grid = np.indices(src_ct.shape, dtype=np.float32)
    for dim in range(3):
        local_grid[dim] = (local_grid[dim] + 0.5) * scale_to_original[dim] - 0.5 + float(mins[dim])
    coord_src = np.moveaxis(local_grid, 0, -1)
    coord_patch[dst_slices[0], dst_slices[1], dst_slices[2]] = coord_src[src_slices[0], src_slices[1], src_slices[2]]

    if norm_mode != "robust_minmax":
        ct_patch = np.clip(ct_patch, -200.0, 1000.0)
    patch = normalize_stage1_patch(
        ct_patch,
        mask_patch,
        norm_mode=norm_mode,
        zscore_scale=zscore_scale,
        foreground_floor=foreground_floor,
    )
    return torch.from_numpy(patch[None, ...]).float(), mask_patch, coord_patch


def _module_by_name(model, name):
    modules = dict(model.named_modules())
    if name not in modules:
        available = ", ".join(modules.keys())
        raise ValueError(f"Layer '{name}' not found. Available layers: {available}")
    return modules[name]


def find_last_conv3d(model):
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            last_name = name
            last_module = module
    if last_module is None:
        raise ValueError("Could not find a Conv3d layer for CAM.")
    return last_name, last_module


def compute_cam(model, input_tensor, target_class=None, method="gradcam", target_layer=None):
    if input_tensor.ndim == 4:
        input_tensor = input_tensor.unsqueeze(0)
    if input_tensor.ndim != 5:
        raise ValueError(f"Expected input shape [B, C, D, H, W] or [C, D, H, W], got {tuple(input_tensor.shape)}")
    if input_tensor.size(0) != 1:
        raise ValueError("CAM computation currently expects a single vertebra patch.")

    device = next(model.parameters()).device
    x = input_tensor.to(device).float()
    if target_layer is None:
        _, layer = find_last_conv3d(model)
    elif isinstance(target_layer, tuple):
        layer = target_layer[1]
    else:
        layer = _module_by_name(model, target_layer)

    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    fh = layer.register_forward_hook(forward_hook)
    bh = layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        probabilities = torch.softmax(logits.detach(), dim=1)
        pred = int(probabilities.argmax(dim=1).item())
        target = pred if target_class is None else int(target_class)
        if target < 0 or target >= logits.shape[1]:
            raise ValueError(f"target_class {target} is outside model output range 0..{logits.shape[1] - 1}")

        score = logits[:, target].sum()
        score.backward(retain_graph=False)

        acts = activations["value"].detach()
        grads = gradients["value"].detach()
        method_key = method.lower().replace("_", "").replace("-", "").replace("+", "p")

        if method_key == "gradcam":
            weights = grads.mean(dim=(2, 3, 4), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
        elif method_key in {"gradcampp", "gradcamplusplus"}:
            grads2 = grads.pow(2)
            grads3 = grads.pow(3)
            denom = 2.0 * grads2 + (acts * grads3).sum(dim=(2, 3, 4), keepdim=True)
            alphas = grads2 / torch.clamp(denom, min=1e-8)
            weights = (alphas * F.relu(grads)).sum(dim=(2, 3, 4), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
        elif method_key == "layercam":
            cam = (F.relu(grads) * acts).sum(dim=1, keepdim=True)
        else:
            raise ValueError("method must be one of: gradcam, gradcam++, layercam")

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0, 0]
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / torch.clamp(cam_max - cam_min, min=1e-8)

        return CAMResult(
            logits=logits.detach().cpu(),
            probabilities=probabilities.cpu(),
            predicted_class=pred,
            target_class=target,
            cam=cam.detach().cpu().numpy().astype(np.float32),
        )
    finally:
        fh.remove()
        bh.remove()


def cam_point_actor(cam, mask=None, threshold=0.35, max_points=25000, point_size=4.0, cmap="jet"):
    from vedo import Points

    cam = np.asarray(cam, dtype=np.float32)
    keep = cam >= float(threshold)
    if mask is not None:
        keep &= np.asarray(mask).astype(bool)
    if not np.any(keep):
        keep = cam >= float(np.quantile(cam, 0.95))
        if mask is not None:
            keep &= np.asarray(mask).astype(bool)

    ijk = np.argwhere(keep)
    values = cam[keep]
    if len(ijk) > max_points:
        order = np.argsort(values)[-int(max_points):]
        ijk = ijk[order]
        values = values[order]

    pts = Points(ijk.astype(np.float32), r=float(point_size))
    try:
        pts.cmap(cmap, values)
    except Exception:
        pts.c("red")
    pts.alpha(0.85)
    return pts


def show_cam_overlay(
    patch,
    cam,
    mask=None,
    title="Stage 1 CAM",
    cam_threshold=0.35,
    max_points=25000,
    point_size=4.0,
    cam_cmap="jet",
):
    from vedo import Text2D, Volume, show

    patch_np = np.asarray(patch.detach().cpu() if torch.is_tensor(patch) else patch, dtype=np.float32)
    if patch_np.ndim == 4:
        patch_np = patch_np[0]
    mask_np = np.asarray(mask if mask is not None else patch_np > 0, dtype=bool)

    actors = []
    if np.any(mask_np):
        surf = Volume(mask_np.astype(np.uint8)).isosurface(0.5).c("lightgray").alpha(0.22)
        actors.append(surf)
    actors.append(
        cam_point_actor(
            cam,
            mask=mask_np,
            threshold=cam_threshold,
            max_points=max_points,
            point_size=point_size,
            cmap=cam_cmap,
        )
    )
    actors.append(Text2D(str(title), pos="top-left", s=0.8))
    show(*actors, axes=1, bg="black", bg2="gray3", title=str(title), interactive=True)


def full_volume_cam_point_actor(cam_points, threshold=None, max_points=250000, point_size=5.0, cmap="jet"):
    from vedo import Points

    coords_parts = []
    value_parts = []
    for item in cam_points:
        coords = np.asarray(item["coords"], dtype=np.float32)
        values = np.asarray(item["values"], dtype=np.float32)
        keep = np.ones(values.shape, dtype=bool) if threshold is None else values >= float(threshold)
        if np.any(keep):
            coords_parts.append(coords[keep])
            value_parts.append(values[keep])

    if not coords_parts:
        for item in cam_points:
            coords = np.asarray(item["coords"], dtype=np.float32)
            values = np.asarray(item["values"], dtype=np.float32)
            if len(values) == 0:
                continue
            cutoff = float(np.quantile(values, 0.95))
            keep = values >= cutoff
            coords_parts.append(coords[keep])
            value_parts.append(values[keep])

    if not coords_parts:
        return None

    coords = np.concatenate(coords_parts, axis=0)
    values = np.concatenate(value_parts, axis=0)
    if len(coords) > max_points:
        sample_idx = np.linspace(0, len(coords) - 1, int(max_points), dtype=np.int64)
        coords = coords[sample_idx]
        values = values[sample_idx]

    pts = Points(coords, r=float(point_size))
    try:
        pts.cmap(cmap, values)
    except Exception:
        pts.c("red")
    pts.alpha(0.95)
    return pts


def show_full_ct_cam_overlay(
    ct,
    seg,
    cam_points,
    title="SINS Stage 1 CAMs",
    cam_threshold=None,
    max_points=250000,
    point_size=5.0,
    cam_cmap="jet",
    ct_alpha=0.06,
    ct_wl=300.0,
    ct_ww=1500.0,
    show_seg_context=False,
):
    from vedo import Text2D, Volume, show

    ct_np = np.asarray(ct, dtype=np.float32)
    seg_np = np.asarray(seg)
    wl = float(ct_wl)
    ww = max(float(ct_ww), 1.0)
    ct_vis = np.clip(ct_np, wl - ww / 2.0, wl + ww / 2.0)

    actors = [
        Volume(ct_vis)
        .cmap("bone")
        .alpha([0.0, 0.0, float(ct_alpha) * 0.4, float(ct_alpha)])
    ]

    if show_seg_context and np.any(seg_np > 0):
        try:
            actors.append(Volume((seg_np > 0).astype(np.uint8)).isosurface(0.5).c("lightgray").alpha(0.16))
        except Exception:
            pass

    cam_actor = full_volume_cam_point_actor(
        cam_points,
        threshold=cam_threshold,
        max_points=max_points,
        point_size=point_size,
        cmap=cam_cmap,
    )
    if cam_actor is not None:
        actors.append(cam_actor)

    actors.append(Text2D(str(title), pos="top-left", s=0.75))
    show(*actors, axes=1, bg="black", bg2="gray3", title=str(title).splitlines()[0], interactive=True)


def latest_checkpoint(root="output/stage1/4_class", filename="best.pth"):
    root = Path(root)
    paths = sorted(root.glob(f"*/{filename}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths and filename == "best.pth":
        paths = sorted(root.glob("*/last.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found under {root}")
    return paths[0]
