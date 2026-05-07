from datetime import date
from dataclasses import dataclass

import numpy as np
from skimage import morphology
import xarray as xr

from .data_fetcher import FetchConfig, fetch_least_cloud_sentinel2
from .prithvi_segmenter import PrithviConfig, PrithviFloodSegmenter
from .report import ResilienceReport, compute_pixel_area_sqm, flooded_area_sqkm


@dataclass
class AnalysisArtifacts:
    report: ResilienceReport
    composite: xr.Dataset
    mask: np.ndarray
    backend: str
    backend_notes: str


def _clean_flood_mask(mask: np.ndarray) -> np.ndarray:
    """
    Reduce salt-and-pepper noise and tiny artifacts in binary flood masks.
    """
    binary = mask.astype(bool)
    binary = morphology.binary_opening(binary, morphology.disk(1))
    binary = morphology.binary_closing(binary, morphology.disk(1))
    binary = morphology.remove_small_objects(binary, min_size=36, connectivity=2)
    binary = morphology.remove_small_holes(binary, area_threshold=64)
    return binary.astype(np.uint8)


def run_resilience_analysis(
    event_name: str,
    bbox: list[float],
    start_date: date,
    end_date: date,
    model_id: str = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11",
) -> AnalysisArtifacts:
    fetch_cfg = FetchConfig(bbox=bbox, start_date=start_date, end_date=end_date)
    composite = fetch_least_cloud_sentinel2(fetch_cfg)

    segmenter = PrithviFloodSegmenter(PrithviConfig(model_id=model_id))
    raw_mask = segmenter.predict_flood_mask(composite)
    mask = _clean_flood_mask(raw_mask)

    pixel_area = compute_pixel_area_sqm(composite)
    area_sqkm, flood_pixels = flooded_area_sqkm(mask, pixel_area)
    report = ResilienceReport(
        event_name=event_name,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        flooded_area_sqkm=area_sqkm,
        flood_pixel_count=flood_pixels,
        pixel_area_sqm=pixel_area,
        notes=(
            "Least-cloud Sentinel-2 composite from Planetary Computer + "
            f"segmentation backend={segmenter.backend}. {segmenter.backend_notes}"
        ),
    )
    return AnalysisArtifacts(
        report=report,
        composite=composite,
        mask=mask,
        backend=segmenter.backend,
        backend_notes=segmenter.backend_notes,
    )


def run_resilience_pipeline(
    event_name: str,
    bbox: list[float],
    start_date: date,
    end_date: date,
    model_id: str = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11",
) -> ResilienceReport:
    artifacts = run_resilience_analysis(
        event_name=event_name,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        model_id=model_id,
    )
    return artifacts.report

