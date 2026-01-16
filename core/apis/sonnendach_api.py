# -*- coding: utf-8 -*-
"""core/apis/sonnendach_api.py (patched - minimal)

This module is intentionally minimal and robust.
Kept features (Phase 1):
- Address autocomplete via GeoAdmin SearchServer (returns LV95 x/y)
- Fetch a Sonnendach roof plane by its FeatureId (manual workflow)

Dropped on purpose:
- Any automatic roof-plane discovery (EGID/GWR/identify/find). Too fragile for now.

Rationale:
- Autocomplete + auto-zoom on geo.admin is handled in UI.
- Roof selection is done by user on geo.admin; we import by FeatureId.
"""

from __future__ import annotations

from typing import Any, Dict, List

import requests


# Sonnendach roof suitability layer (roof planes)
LAYER_SONNENDACH = "ch.bfe.solarenergie-eignung-daecher"

# GeoAdmin SearchServer (locations)
SEARCH_BASE = "https://api3.geo.admin.ch/rest/services/ech/SearchServer"

# Feature fetch by id (layer resource)
FEATURE_BASE = "https://api3.geo.admin.ch/rest/services/ech/MapServer"


def _http_get_json(url: str, params: Dict[str, Any], timeout_s: int = 15) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise ValueError(f"Réponse JSON inattendue (type={type(data)}). URL={url}")
    return data


def search_addresses(query: str, *, limit: int = 10, lang: str = "fr", sr: int = 2056) -> List[Dict[str, Any]]:
    """Search Swiss addresses via GeoAdmin SearchServer.

    Returns list of candidates:
      - label: str
      - x, y (LV95 if sr=2056)
      - sr: int

    Notes:
    - SearchServer may return x/y swapped in some cases; we normalize for LV95.
    """
    q = (query or "").strip()
    if len(q) < 3:
        return []

    params = {
        "searchText": q,
        "type": "locations",
        "origins": "address",
        "lang": lang,
        "limit": int(limit),
        "sr": int(sr),
    }

    data = _http_get_json(SEARCH_BASE, params=params, timeout_s=15)
    results = data.get("results") or []

    out: List[Dict[str, Any]] = []
    for r in results:
        attrs = (r or {}).get("attrs") or {}

        x = attrs.get("x") if attrs.get("x") is not None else (r or {}).get("x")
        y = attrs.get("y") if attrs.get("y") is not None else (r or {}).get("y")

        # fallback lon/lat (rare)
        if x is None and attrs.get("lon") is not None:
            x = attrs.get("lon")
        if y is None and attrs.get("lat") is not None:
            y = attrs.get("lat")

        label = attrs.get("label") or attrs.get("title") or (r or {}).get("text") or ""
        if x is None or y is None or not label:
            continue

        # Normalize LV95: E ~ 2.4..2.9 Mio, N ~ 1.0..1.3 Mio
        try:
            fx = float(x)
            fy = float(y)
            if int(sr) == 2056 and fx < 2_000_000 and fy > 2_000_000:
                fx, fy = fy, fx
            x, y = fx, fy
        except Exception:
            # keep raw if parsing fails
            pass

        try:
            out.append({
                "label": str(label),
                "x": float(x),
                "y": float(y),
                "sr": int(sr),
                "raw": r,
            })
        except Exception:
            continue

    return out


def fetch_roof_by_feature_id(feature_id: int | str, *, lang: str = "fr") -> Dict[str, Any]:
    """Fetch Sonnendach roof-plane attributes by FeatureId.

    The FeatureId is obtained by the user from geo.admin.
    """
    fid = str(feature_id).strip()
    if not fid:
        raise ValueError("feature_id vide")

    url = f"{FEATURE_BASE}/{LAYER_SONNENDACH}/{fid}"
    params = {"returnGeometry": "false", "lang": lang}
    data = _http_get_json(url, params=params, timeout_s=15)

    # Common response shape
    if "feature" in data:
        attrs = (data["feature"] or {}).get("attributes") or {}
        if attrs:
            return attrs

    # Alternate shape
    feats = data.get("features") or []
    if feats:
        attrs = (feats[0] or {}).get("attributes") or {}
        if attrs:
            return attrs
        props = (feats[0] or {}).get("properties") or {}
        if props:
            return props

    raise ValueError(f"Aucun attribut renvoyé pour featureId={fid}.")
