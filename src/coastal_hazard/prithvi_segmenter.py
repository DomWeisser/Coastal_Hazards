from dataclasses import dataclass

import numpy as np
import torch
import xarray as xr
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


@dataclass
class PrithviConfig:
    model_id: str = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11"
    threshold: float = 0.5
    ndwi_threshold: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PrithviFloodSegmenter:
    """
    Wrapper around a Prithvi-based segmentation checkpoint.
    Use a flood fine-tuned model ID/checkpoint for production quality results.
    """

    def __init__(self, config: PrithviConfig):
        self.config = config
        self.backend = "prithvi_transformers"
        self.backend_notes = "Prithvi model loaded through Hugging Face Transformers."
        self.processor = None
        self.model = None
        try:
            self.processor = AutoImageProcessor.from_pretrained(config.model_id)
            self.model = AutoModelForSemanticSegmentation.from_pretrained(config.model_id)
            self.model.to(config.device)
            self.model.eval()
        except Exception as exc:
            # Model card artifacts currently follow TerraTorch/MMseg format. Fallback keeps API usable.
            self.backend = "ndwi_fallback"
            self.backend_notes = (
                "Prithvi artifacts were not compatible with Transformers auto-loader; "
                f"using NDWI fallback mask. Root error: {exc}"
            )

    def predict_flood_mask(self, composite_ds: xr.Dataset) -> np.ndarray:
        """
        Accepts composite with B03 and B08 bands.
        Returns binary mask: 1=flood/water, 0=non-flood.
        """
        green = composite_ds["B03"].values
        nir = composite_ds["B08"].values
        green = np.nan_to_num(green, nan=0.0).astype(np.float32)
        nir = np.nan_to_num(nir, nan=0.0).astype(np.float32)

        if self.backend == "ndwi_fallback":
            # NDWI=(G-NIR)/(G+NIR): positive values commonly indicate water.
            denom = green + nir
            denom[denom == 0.0] = 1e-6
            ndwi = (green - nir) / denom
            return (ndwi > self.config.ndwi_threshold).astype(np.uint8)

        # Stack to pseudo-RGB expected by many processors.
        rgb_like = np.stack([green, nir, green], axis=-1)

        inputs = self.processor(images=rgb_like, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits
            probs = torch.sigmoid(logits)
            flood_prob = probs[:, 0, :, :]
            mask = (flood_prob > self.config.threshold).to(torch.uint8)
        return mask.squeeze(0).cpu().numpy()

