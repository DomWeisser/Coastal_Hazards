from datetime import date
from datetime import timedelta

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import run_resilience_analysis, run_resilience_pipeline
from .visualization import save_before_after_maps, save_flood_maps

app = FastAPI(title="Global Coastal Hazard AI")


class ResilienceRequest(BaseModel):
    event_name: str = Field(default="Cyclone Remal Bangladesh")
    bbox: list[float] = Field(
        default=[88.5, 21.5, 89.5, 22.8],
        description="[min_lon, min_lat, max_lon, max_lat]",
        min_length=4,
        max_length=4,
    )
    start_date: date = Field(default=date(2024, 5, 20))
    end_date: date = Field(default=date(2024, 5, 31))
    model_id: str = Field(default="ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11")
    output_dir: str = Field(default="outputs")
    output_prefix: str = Field(default="remal_2024")
    include_before_after: bool = Field(default=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/resilience-report")
def resilience_report(req: ResilienceRequest) -> dict:
    report = run_resilience_pipeline(
        event_name=req.event_name,
        bbox=req.bbox,
        start_date=req.start_date,
        end_date=req.end_date,
        model_id=req.model_id,
    )
    return report.to_dict()


@app.post("/resilience-maps")
def resilience_maps(req: ResilienceRequest) -> dict:
    post = run_resilience_analysis(
        event_name=req.event_name,
        bbox=req.bbox,
        start_date=req.start_date,
        end_date=req.end_date,
        model_id=req.model_id,
    )
    map_paths = save_flood_maps(
        ds=post.composite,
        mask=post.mask,
        output_dir=req.output_dir,
        prefix=req.output_prefix,
    )

    response = {
        "report": post.report.to_dict(),
        "map_paths": map_paths,
    }

    if req.include_before_after:
        duration_days = max((req.end_date - req.start_date).days + 1, 1)
        pre_end = req.start_date - timedelta(days=1)
        pre_start = pre_end - timedelta(days=duration_days - 1)

        pre = run_resilience_analysis(
            event_name=f"{req.event_name} pre-event",
            bbox=req.bbox,
            start_date=pre_start,
            end_date=pre_end,
            model_id=req.model_id,
        )
        pre_post_paths = save_before_after_maps(
            pre_mask=pre.mask,
            post_mask=post.mask,
            output_dir=req.output_dir,
            prefix=req.output_prefix,
        )
        response["before_after_window"] = {
            "pre_start_date": pre_start.isoformat(),
            "pre_end_date": pre_end.isoformat(),
            "post_start_date": req.start_date.isoformat(),
            "post_end_date": req.end_date.isoformat(),
        }
        response["before_after_map_paths"] = pre_post_paths

    return response

