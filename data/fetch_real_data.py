"""Real-world data anchoring for the anti-poaching patrol framework.

Fetches and processes publicly available occurrence records for
Diceros bicornis (black rhinoceros) from the Global Biodiversity
Information Facility (GBIF) API and maps them onto the Etosha
computational grid.

The occurrence density grid is used as a spatial prior in the WPP
field: cells with documented rhino sightings receive a modest upward
weight, anchoring the modelled priority surface to observed animal
distributions.

Data source:
    GBIF Occurrence API — https://www.gbif.org/developer/occurrence
    Taxon: Diceros bicornis (Linnaeus, 1758), key 5220111
    License: CC-BY 4.0 (GBIF mediated datasets)

Citation:
    GBIF.org (2026) GBIF Occurrence Download.
    https://doi.org/10.15468/dl.XXXXXXX  [replace with actual DOI]
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# Etosha National Park approximate bounding box (WGS84)
ETOSHA_LAT_MIN: float = -19.25
ETOSHA_LAT_MAX: float = -18.20
ETOSHA_LON_MIN: float = 15.60
ETOSHA_LON_MAX: float = 16.55

# Southern Africa bounding box for initial GBIF filter
SOUTH_AFRICA_LAT_MIN: float = -30.0
SOUTH_AFRICA_LAT_MAX: float = -16.0
SOUTH_AFRICA_LON_MIN: float = 10.0
SOUTH_AFRICA_LON_MAX: float = 26.0

GBIF_TAXON_KEY: int = 5220111  # Diceros bicornis
GBIF_BASE_URL: str = "https://api.gbif.org/v1/occurrence/search"
CACHE_PATH: Path = Path(__file__).parent.parent / "data" / "gbif_rhino_cache.json"


@dataclass
class OccurrenceRecord:
    """Single georeferenced occurrence record."""
    latitude: float
    longitude: float
    year: int
    country: str


def _fetch_page(offset: int = 0, limit: int = 300) -> dict:
    """Fetches one page of GBIF occurrence results."""
    params = (
        f"taxonKey={GBIF_TAXON_KEY}"
        f"&hasCoordinate=true"
        f"&limit={limit}"
        f"&offset={offset}"
    )
    url = f"{GBIF_BASE_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "anti-poaching-patrol-framework/1.0"})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode())


def fetch_gbif_occurrences(
    use_cache: bool = True,
    max_records: int = 600,
) -> List[OccurrenceRecord]:
    """Downloads Diceros bicornis occurrence records from GBIF.

    Filters to the southern Africa region and returns structured records.
    Results are cached to ``data/gbif_rhino_cache.json`` to avoid
    repeated API calls.

    Args:
        use_cache: If True, read from cache file when available.
        max_records: Maximum total records to fetch across pages.

    Returns:
        List of OccurrenceRecord within the southern Africa bounding box.
    """
    if use_cache and CACHE_PATH.exists():
        with CACHE_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return [OccurrenceRecord(**r) for r in raw]

    records: List[OccurrenceRecord] = []
    offset = 0
    limit = 300

    while offset < max_records:
        try:
            page = _fetch_page(offset=offset, limit=min(limit, max_records - offset))
        except (urllib.error.URLError, OSError):
            break

        results = page.get("results", [])
        if not results:
            break

        for item in results:
            lat = item.get("decimalLatitude")
            lon = item.get("decimalLongitude")
            if lat is None or lon is None:
                continue
            # Filter to southern Africa
            if not (SOUTH_AFRICA_LAT_MIN <= lat <= SOUTH_AFRICA_LAT_MAX
                    and SOUTH_AFRICA_LON_MIN <= lon <= SOUTH_AFRICA_LON_MAX):
                continue
            records.append(OccurrenceRecord(
                latitude=float(lat),
                longitude=float(lon),
                year=int(item.get("year", 0)),
                country=str(item.get("countryCode", "")),
            ))

        if len(results) < limit:
            break
        offset += limit

    # Cache results
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as fh:
        json.dump([r.__dict__ for r in records], fh, indent=2)

    return records


def records_to_grid(
    records: List[OccurrenceRecord],
    grid_side: int,
    lat_min: float = ETOSHA_LAT_MIN,
    lat_max: float = ETOSHA_LAT_MAX,
    lon_min: float = ETOSHA_LON_MIN,
    lon_max: float = ETOSHA_LON_MAX,
    sigma_cells: float = 1.5,
) -> NDArray[np.float64]:
    """Maps occurrence records onto a grid and applies Gaussian smoothing.

    Points outside the bounding box are discarded. The resulting grid
    represents a kernel-smoothed occurrence density that can be used as
    a spatial prior in the WPP field.

    Args:
        records: GBIF occurrence records.
        grid_side: Number of grid cells per side (matches topology.grid_side).
        lat_min, lat_max, lon_min, lon_max: Bounding box in WGS84 degrees.
        sigma_cells: Gaussian smoothing radius in grid cells.

    Returns:
        2D array of shape (grid_side, grid_side), normalised to [0, 1].
    """
    density = np.zeros((grid_side, grid_side), dtype=float)

    for rec in records:
        if not (lat_min <= rec.latitude <= lat_max and lon_min <= rec.longitude <= lon_max):
            continue
        # Convert to grid indices (y = row = south-to-north, x = col = west-to-east)
        col = int((rec.longitude - lon_min) / (lon_max - lon_min) * (grid_side - 1))
        row = int((lat_max - rec.latitude) / (lat_max - lat_min) * (grid_side - 1))
        col = int(np.clip(col, 0, grid_side - 1))
        row = int(np.clip(row, 0, grid_side - 1))
        density[row, col] += 1.0

    # Gaussian kernel smoothing
    if density.max() > 0 and sigma_cells > 0:
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=sigma_cells)

    # Normalise to [0, 1]
    max_val = float(density.max())
    if max_val > 0:
        density /= max_val

    return density


def get_occurrence_prior(
    grid_side: int,
    use_cache: bool = True,
) -> Tuple[NDArray[np.float64], int]:
    """Convenience function: fetch GBIF data and return grid prior.

    Args:
        grid_side: Grid dimension from topology.
        use_cache: Whether to use cached data.

    Returns:
        Tuple of (occurrence_prior_grid, n_records_in_etosha_bbox).
    """
    records = fetch_gbif_occurrences(use_cache=use_cache)
    etosha_records = [
        r for r in records
        if ETOSHA_LAT_MIN <= r.latitude <= ETOSHA_LAT_MAX
        and ETOSHA_LON_MIN <= r.longitude <= ETOSHA_LON_MAX
    ]
    prior = records_to_grid(records, grid_side=grid_side)
    return prior, len(etosha_records)


def gbif_summary(records: List[OccurrenceRecord]) -> dict:
    """Returns summary statistics for the fetched occurrence dataset."""
    if not records:
        return {"n_total": 0, "n_namibia": 0, "n_etosha_bbox": 0,
                "year_min": 0, "year_max": 0}
    years = [r.year for r in records if r.year > 0]
    namibia = [r for r in records if r.country == "NA"]
    etosha = [
        r for r in records
        if ETOSHA_LAT_MIN <= r.latitude <= ETOSHA_LAT_MAX
        and ETOSHA_LON_MIN <= r.longitude <= ETOSHA_LON_MAX
    ]
    return {
        "n_total": len(records),
        "n_namibia": len(namibia),
        "n_etosha_bbox": len(etosha),
        "year_min": int(min(years)) if years else 0,
        "year_max": int(max(years)) if years else 0,
    }
