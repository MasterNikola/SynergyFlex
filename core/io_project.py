# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 09:48:26 2025

@author: vujic
"""

# core/io_project.py
import json
from typing import Dict, Any


def export_project_dict(project: Dict[str, Any]) -> bytes:
    """
    Transforme le dict projet en bytes JSON.
    Utilisé par st.download_button.
    """
    return json.dumps(project, ensure_ascii=False, indent=2).encode("utf-8")


def import_project_from_json(uploaded_file) -> Dict[str, Any]:
    """
    Charge un projet depuis un fichier JSON uploadé.
    """
    data = uploaded_file.read().decode("utf-8")
    project = json.loads(data)
    return project
