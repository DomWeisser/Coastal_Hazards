from pathlib import Path

import imageio.v3 as iio
import numpy as np
import xarray as xr


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    p2 = float(np.percentile(arr, 2))
    p98 = float(np.percentile(arr, 98))
    if p98 <= p2:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)


def _as_uint8(img_01: np.ndarray) -> np.ndarray:
    return (np.clip(img_01, 0.0, 1.0) * 255.0).astype(np.uint8)


def _base_gray_rgb(ds: xr.Dataset) -> np.ndarray:
    red = _normalize_01(np.asarray(ds["B04"].values))
    green = _normalize_01(np.asarray(ds["B03"].values))
    blue = _normalize_01(np.asarray(ds["B02"].values))
    rgb = np.stack([red, green, blue], axis=-1)
    # Mild gamma to brighten low-reflectance wetlands.
    gamma = 0.85
    return np.clip(np.power(rgb, gamma), 0.0, 1.0)


def _ndwi(ds: xr.Dataset) -> np.ndarray:
    green = np.nan_to_num(np.asarray(ds["B03"].values), nan=0.0)
    nir = np.nan_to_num(np.asarray(ds["B08"].values), nan=0.0)
    denom = green + nir
    denom[denom == 0.0] = 1e-6
    return (green - nir) / denom


def _mndwi(ds: xr.Dataset) -> np.ndarray:
    green = np.nan_to_num(np.asarray(ds["B03"].values), nan=0.0)
    swir = np.nan_to_num(np.asarray(ds["B11"].values), nan=0.0)
    denom = green + swir
    denom[denom == 0.0] = 1e-6
    return (green - swir) / denom


def save_flood_maps(
    ds: xr.Dataset,
    mask: np.ndarray,
    output_dir: str,
    prefix: str,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = _base_gray_rgb(ds)
    mask_bin = (mask > 0).astype(np.uint8)
    overlay = base.copy()
    flood_color = np.array([1.0, 0.1, 0.1], dtype=np.float32)
    alpha = 0.55
    overlay[mask_bin == 1] = (1.0 - alpha) * overlay[mask_bin == 1] + alpha * flood_color

    ndwi = _ndwi(ds)
    ndwi_01 = np.clip((ndwi + 1.0) / 2.0, 0.0, 1.0)
    ndwi_rgb = np.stack([np.zeros_like(ndwi_01), ndwi_01, 1.0 - ndwi_01], axis=-1)
    mndwi = _mndwi(ds)
    mndwi_01 = np.clip((mndwi + 1.0) / 2.0, 0.0, 1.0)
    mndwi_rgb = np.stack([np.zeros_like(mndwi_01), mndwi_01, 1.0 - mndwi_01], axis=-1)

    mask_rgb = np.zeros((mask_bin.shape[0], mask_bin.shape[1], 3), dtype=np.float32)
    mask_rgb[mask_bin == 1] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    side_by_side = np.concatenate([base, overlay], axis=1)

    files = {
        "base_map": out / f"{prefix}_base_map.png",
        "flood_mask": out / f"{prefix}_flood_mask.png",
        "flood_overlay": out / f"{prefix}_flood_overlay.png",
        "ndwi_map": out / f"{prefix}_ndwi_map.png",
        "mndwi_map": out / f"{prefix}_mndwi_map.png",
        "side_by_side": out / f"{prefix}_side_by_side.png",
    }
    iio.imwrite(files["base_map"], _as_uint8(base))
    iio.imwrite(files["flood_mask"], _as_uint8(mask_rgb))
    iio.imwrite(files["flood_overlay"], _as_uint8(overlay))
    iio.imwrite(files["ndwi_map"], _as_uint8(ndwi_rgb))
    iio.imwrite(files["mndwi_map"], _as_uint8(mndwi_rgb))
    iio.imwrite(files["side_by_side"], _as_uint8(side_by_side))
    return {k: str(v) for k, v in files.items()}


def save_before_after_maps(
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    output_dir: str,
    prefix: str,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pre = (pre_mask > 0).astype(np.uint8)
    post = (post_mask > 0).astype(np.uint8)

    pre_rgb = np.stack([pre, pre, pre], axis=-1).astype(np.float32)
    post_rgb = np.stack([post, post, post], axis=-1).astype(np.float32)
    before_after = np.concatenate([pre_rgb, post_rgb], axis=1)

    new_flood = (pre == 0) & (post == 1)
    persistent = (pre == 1) & (post == 1)
    receded = (pre == 1) & (post == 0)
    flood_only_post = (post == 1) & (pre == 0)

    change = np.zeros((post.shape[0], post.shape[1], 3), dtype=np.float32)
    # Legend: red=new flood; purple=persistent water; blue=receded water.
    change[new_flood] = np.array([1.0, 0.1, 0.1], dtype=np.float32)
    change[persistent] = np.array([0.8, 0.2, 1.0], dtype=np.float32)
    change[receded] = np.array([0.1, 0.3, 1.0], dtype=np.float32)

    flood_only = np.zeros((post.shape[0], post.shape[1], 3), dtype=np.float32)
    flood_only[flood_only_post] = np.array([1.0, 0.1, 0.1], dtype=np.float32)

    files = {
        "before_after_masks": out / f"{prefix}_before_after_masks.png",
        "flood_change_map": out / f"{prefix}_flood_change_map.png",
        "flood_only_extent_map": out / f"{prefix}_flood_only_extent_map.png",
    }
    iio.imwrite(files["before_after_masks"], _as_uint8(before_after))
    iio.imwrite(files["flood_change_map"], _as_uint8(change))
    iio.imwrite(files["flood_only_extent_map"], _as_uint8(flood_only))
    return {k: str(v) for k, v in files.items()}

