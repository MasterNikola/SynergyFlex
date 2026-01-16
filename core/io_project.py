# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 09:48:26 2025

@author: vujic
"""

# core/io_project.py
import json
from typing import Dict, Any
from copy import deepcopy

def export_project_dict(project, include_results=False, include_timeseries=False):
    """
    Retourne TOUJOURS des bytes JSON (jamais None).
    """
    payload = deepcopy(project) if project is not None else {}

    # Config only
    if not include_results:
        payload.pop("results", None)
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    # Results summary / full
    results = payload.get("results") or {}

    # Timeseries store (hybride)
    if not include_timeseries:
        results.pop("timeseries_store", None)

    # Retirer profiles dans flows (au cas où)
    flows_root = (results.get("flows") or {})
    bats = (flows_root.get("batiments") or {})

    try:
        for _bat_id, bat_data in bats.items():
            if isinstance(bat_data, dict):
                blocks = bat_data.get("flow_blocks") or []
            elif isinstance(bat_data, list):
                blocks = bat_data
            else:
                continue

            for b in blocks:
                if isinstance(b, dict):
                    b.pop("profiles", None)
    except Exception as e:
        # On n'échoue pas l'export pour un nettoyage "best effort"
        results.setdefault("export_warnings", [])
        results["export_warnings"].append(f"Flow profiles cleanup skipped: {e}")

    payload["results"] = results
    # ----------------------------------------------------------
    # Make timeseries_by_batiment JSON-serializable (best effort)
    # - Convert pd.Series -> list
    # - Ensure an "index" field exists (ISO strings) when possible
    # This prevents Phase1<->Phase2 roundtrips from losing/invalidating timeseries.
    # ----------------------------------------------------------
    def _make_ts_jsonable(ts_by_bat: dict) -> dict:
        try:
            import pandas as pd
        except Exception:
            pd = None
    
        if not isinstance(ts_by_bat, dict):
            return ts_by_bat
    
        for bat_id, bat_data in ts_by_bat.items():
            if not isinstance(bat_data, dict):
                continue
    
            measured = bat_data.get("measured")
            if not isinstance(measured, dict):
                continue
    
            # If we have at least one pd.Series, use its index as reference
            ref_index = None
            if pd is not None:
                for v in measured.values():
                    if isinstance(v, pd.Series):
                        ref_index = v.index
                        break
    
            # Ensure an exportable index exists
            if bat_data.get("index") in (None, []) and ref_index is not None:
                try:
                    # Convert datetime-like index to ISO strings
                    idx = pd.to_datetime(ref_index, errors="coerce")
                    if not idx.isna().all():
                        bat_data["index"] = [x.isoformat() for x in idx.to_pydatetime()]
                    else:
                        bat_data["index"] = list(range(len(ref_index)))
                except Exception:
                    bat_data["index"] = list(range(len(ref_index)))
    
            # Convert series -> list (keep other types unchanged)
            if pd is not None:
                for k, v in list(measured.items()):
                    if isinstance(v, pd.Series):
                        measured[k] = [float(x) if x is not None else 0.0 for x in v.fillna(0.0).tolist()]
    
        return ts_by_bat

        # ----------------------------------------------------------
    # Timeseries export policy
    # - timeseries_store: always removed unless include_timeseries=True
    # - timeseries_by_batiment: keep it, but make it JSON-serializable
    #   so Phase1<->Phase2 roundtrips don't break plots.
    # ----------------------------------------------------------
    if not include_timeseries:
        results.pop("timeseries_store", None)

        ts_by_bat = results.get("timeseries_by_batiment")
        if isinstance(ts_by_bat, dict):
            results["timeseries_by_batiment"] = _make_ts_jsonable(ts_by_bat)


    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")




def import_project_from_json(uploaded_file) -> Dict[str, Any]:
    """
    Charge un projet depuis un fichier JSON uploadé.
    """
    data = uploaded_file.read().decode("utf-8")
    project = json.loads(data)
    return project
