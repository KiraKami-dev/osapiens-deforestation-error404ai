"""Convert binary deforestation prediction rasters to GeoJSON (submission pipeline)."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def raster_to_geojson(
    raster_path: str | Path,
    output_path: str | Path | None = None,
    min_area_ha: float = 0.5,
) -> dict:
    """Vectorise a single-band binary GeoTIFF (1 = deforest) to EPSG:4326 features; drop polygons < ``min_area_ha``."""
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.uint8)
        transform = src.transform
        crs = src.crs

    if data.sum() == 0:
        raise ValueError(
            f"No deforestation pixels (value=1) found in {raster_path}. "
            "Ensure the raster has been binarised before calling this function."
        )

    polygons = [
        shape(geom)
        for geom, value in shapes(data, mask=data, transform=transform)
        if value == 1
    ]

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf = gdf.to_crs("EPSG:4326")

    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    gdf = gdf[gdf_utm.area / 10_000 >= min_area_ha].reset_index(drop=True)

    if gdf.empty:
        raise ValueError(
            f"All polygons are smaller than min_area_ha={min_area_ha} ha. "
            "Lower the threshold or check your prediction raster."
        )

    gdf["time_step"] = None

    geojson = json.loads(gdf.to_json())

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f)

    return geojson
