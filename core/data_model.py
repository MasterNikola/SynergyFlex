# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 09:42:43 2025

@author: vujic
"""

# core/data_model.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class Ouvrage:
    id: str
    batiment_id: str
    type_ouvrage: str        # ex. "PAC", "Batterie", "PV", "UTES", "PCM"
    puissance_kw: float | None = None
    capacite_kwh: float | None = None
    rendement: float | None = None
    meta: Dict[str, Any] | None = None      # infos supplémentaires


@dataclass
class Batiment:
    id: str
    nom: str
    surface_m2: float | None = None
    volume_m3: float | None = None
    meta: Dict[str, Any] | None = None


@dataclass
class Projet:
    nom: str
    batiments: List[Batiment]
    ouvrages: List[Ouvrage]
    params: Dict[str, Any]        # paramètres généraux du projet
    results: Dict[str, Any]       # résultats des calculs (bilans, variantes, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nom": self.nom,
            "batiments": [asdict(b) for b in self.batiments],
            "ouvrages": [asdict(o) for o in self.ouvrages],
            "params": self.params,
            "results": self.results,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Projet":
        bats = [Batiment(**b) for b in d.get("batiments", [])]
        ouvrs = [Ouvrage(**o) for o in d.get("ouvrages", [])]
        return Projet(
            nom=d.get("nom", "Projet sans nom"),
            batiments=bats,
            ouvrages=ouvrs,
            params=d.get("params", {}),
            results=d.get("results", {}),
        )
