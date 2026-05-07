import json
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import planetary_computer
import xarray as xr
from odc.stac import load
from pystac_client import Client


@dataclass
class FetchConfig:
    bbox: list[float]
    start_date: date
    end_date: date
    max_items: int = 40
    resolution: int = 10
    max_total_pixels: int = 4_000_000
    epsg: int = 32646
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    collection: str = "sentinel-2-l2a"


def parse_bbox(bbox_json: str) -> list[float]:
    """
    Parse JSON-style bbox string into [min_lon, min_lat, max_lon, max_lat].
    Example:
      "[88.5, 21.5, 89.5, 22.8]"
    """
    bbox = json.loads(bbox_json)
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox must be a JSON list with four numbers.")
    bbox_floats = [float(v) for v in bbox]
    min_lon, min_lat, max_lon, max_lat = bbox_floats
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat].")
    return bbox_floats


def _date_range(start_date: date, end_date: date) -> str:
    return f"{start_date.isoformat()}/{end_date.isoformat()}"


def _adaptive_resolution_meters(config: FetchConfig) -> int:
    """
    Prevent out-of-memory by increasing pixel size for large AOIs.
    Keeps requested resolution when AOI is small enough.
    """
    min_lon, min_lat, max_lon, max_lat = config.bbox
    center_lat = (min_lat + max_lat) / 2.0

    # Approximate geographic to meters conversion.
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(center_lat))

    width_m = max((max_lon - min_lon) * meters_per_deg_lon, 1.0)
    height_m = max((max_lat - min_lat) * meters_per_deg_lat, 1.0)
    area_m2 = width_m * height_m

    # Pixel size needed so area/pixel_area <= max_total_pixels.
    adaptive = int(np.ceil(np.sqrt(area_m2 / float(config.max_total_pixels))))
    return max(config.resolution, adaptive)


def _assign_cloud_cover_by_time(ds: xr.Dataset, items: list) -> xr.Dataset:
    """
    Map STAC item cloud cover onto dataset time coordinates.
    This is needed because groupby='solar_day' can collapse many items into fewer time steps.
    """
    cloud_by_day: dict[np.datetime64, float] = {}
    for it in items:
        if it.datetime is None:
            continue
        day_key = np.datetime64(it.datetime.date().isoformat())
        cloud = float(it.properties.get("eo:cloud_cover", 100.0))
        # Keep best (lowest cloud) representative for each day.
        if day_key not in cloud_by_day or cloud < cloud_by_day[day_key]:
            cloud_by_day[day_key] = cloud

    time_days = np.array(
        [np.datetime64(np.datetime_as_string(t, unit="D")) for t in ds["time"].values]
    )
    cloud_values = np.array([cloud_by_day.get(day, 100.0) for day in time_days], dtype=np.float32)
    return ds.assign_coords({"eo:cloud_cover": ("time", cloud_values)})


def _prepare_cloud_mask(ds: xr.Dataset) -> xr.DataArray:
    """
    Use Sentinel-2 Scene Classification Layer (SCL) to keep likely clear pixels.
    Clear classes: vegetation, bare soil, water, unclassified.
    """
    scl = ds["SCL"]
    clear_classes = [4, 5, 6, 7]
    clear_mask = xr.zeros_like(scl, dtype=bool)
    for cls in clear_classes:
        clear_mask = clear_mask | (scl == cls)
    return clear_mask


def _least_cloud_cover_composite(ds: xr.Dataset, bands: Iterable[str]) -> xr.Dataset:
    """
    Build a per-pixel least-cloud composite.
    1) Sort observations by scene-level cloud cover.
    2) Keep only clear pixels by SCL.
    3) For each pixel, take first valid value from least-cloudy to most-cloudy image.
    """
    cloud_sorted = ds.sortby(ds["eo:cloud_cover"])
    clear_mask = _prepare_cloud_mask(cloud_sorted)

    composite_vars: dict[str, xr.DataArray] = {}
    for band in bands:
        clear_band = cloud_sorted[band].where(clear_mask)
        stitched = clear_band.isel(time=0)
        for t in range(1, clear_band.sizes["time"]):
            stitched = stitched.combine_first(clear_band.isel(time=t))
        composite_vars[band] = stitched

    return xr.Dataset(composite_vars, attrs=ds.attrs)


def fetch_least_cloud_sentinel2(config: FetchConfig) -> xr.Dataset:
    """
    Fetch Sentinel-2 imagery from Planetary Computer and build least-cloud composite.
    Returns bands used by flood segmentation plus geospatial coordinates.
    """
    catalog = Client.open(config.stac_url, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[config.collection],
        bbox=config.bbox,
        datetime=_date_range(config.start_date, config.end_date),
        limit=config.max_items,
        query={"eo:cloud_cover": {"lt": 95}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No Sentinel-2 scenes found for this bbox/date range.")

    bands = ["B02", "B03", "B04", "B08", "B11", "SCL"]  # RGB, NIR, SWIR, SCL.
    effective_resolution = _adaptive_resolution_meters(config)
    ds = load(
        items,
        bands=bands,
        bbox=config.bbox,
        resolution=effective_resolution,
        crs=f"EPSG:{config.epsg}",
        chunks={"x": 1024, "y": 1024},
        groupby="solar_day",
    )
    if "time" not in ds:
        raise RuntimeError("Loaded dataset does not include a time dimension.")

    if "eo:cloud_cover" not in ds:
        ds = _assign_cloud_cover_by_time(ds, items)

    return _least_cloud_cover_composite(ds, bands=["B02", "B03", "B04", "B08", "B11"])

