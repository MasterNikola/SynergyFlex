# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 11:34:44 2025

@author: infor
"""

# core/nodes_ontology.py


"""
Ontologie des nœuds de Sankey pour SynergyFlex.

Tous les modules (PV, demande, PAC, batterie, etc.)
doivent utiliser ces helpers plutôt que des strings en dur.
"""

# ---- ÉLECTRICITÉ : nœuds "fixes" ----

ELEC_GRID = "ELEC_GRID"
ELEC_LOAD = "ELEC_LOAD"
ELEC_LOSS = "ELEC_LOSS"        # optionnel, pour plus tard

def node_elec_grid() -> str:
    return ELEC_GRID

def node_elec_load() -> str:
    return ELEC_LOAD

def node_elec_loss() -> str:
    return ELEC_LOSS


# ---- ÉLECTRICITÉ : producteurs ----

def node_pv(index: int = 1) -> str:
    """Producteur PV n°index."""
    return f"ELEC_PROD_PV_{index}"

def node_elec_prod(tech: str, index: int = 1) -> str:
    """Autre producteur électrique (CHP, Diesel, etc.)."""
    tech = tech.upper()
    return f"ELEC_PROD_{tech}_{index}"


# ---- ÉLECTRICITÉ : stockage ----

def node_battery(index: int = 1) -> str:
    """Stockage électrique (batterie) n°index."""
    return f"ELEC_STO_BAT_{index}"


# ---- THERMIQUE ----

HEAT_LOAD = "HEAT_LOAD"
HEAT_GRID = "HEAT_GRID"
HEAT_LOSS = "HEAT_LOSS"

def node_heat_load() -> str:
    return HEAT_LOAD

def node_heat_grid() -> str:
    return HEAT_GRID

def node_heat_loss() -> str:
    return HEAT_LOSS

def node_heat_prod(tech: str = "UNKNOWN", index: int = 1) -> str:
    """
    Source de chaleur (chaudière, PAC côté chaleur, etc.).
    Par défaut UNKNOWN pour la demande générique actuelle.
    """
    tech = tech.upper()
    return f"HEAT_PROD_{tech}_{index}"


# ---- DEVICES MULTI-ÉNERGIE (PAC, CHP, etc.) ----

def node_device(tech: str, index: int = 1) -> str:
    tech = tech.upper()
    return f"DEV_{tech}_{index}"

