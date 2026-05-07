from dataclasses import dataclass
from datetime import date

import numpy as np
import xarray as xr


@dataclass
class ResilienceReport:
    event_name: str
    bbox: list[float]
    start_date: date
    end_date: date
    flooded_area_sqkm: float
    flood_pixel_count: int
    pixel_area_sqm: float
    notes: str

    def to_dict(self) -> dict:
        return {
            "event_name": self.event_name,
            "bbox": self.bbox,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "flooded_area_sqkm": round(self.flooded_area_sqkm, 3),
            "flood_pixel_count": self.flood_pixel_count,
            "pixel_area_sqm": round(self.pixel_area_sqm, 3),
            "notes": self.notes,
        }


def compute_pixel_area_sqm(ds: xr.Dataset) -> float:
    """
    Estimate pixel area from coordinate spacing.
    For best results, reproject to a metric CRS before area computation.
    """
    if "x" not in ds.coords or "y" not in ds.coords:
        raise RuntimeError("Dataset must have x and y coordinates.")
    x = ds.coords["x"].values
    y = ds.coords["y"].values
    if len(x) < 2 or len(y) < 2:
        raise RuntimeError("Dataset must have at least 2 x and y coordinates.")
    pixel_width = float(np.abs(x[1] - x[0]))
    pixel_height = float(np.abs(y[1] - y[0]))
    return pixel_width * pixel_height


def flooded_area_sqkm(mask: np.ndarray, pixel_area_sqm: float) -> tuple[float, int]:
    flood_pixels = int(mask.sum())
    area_sqm = flood_pixels * pixel_area_sqm
    return area_sqm / 1_000_000.0, flood_pixels

