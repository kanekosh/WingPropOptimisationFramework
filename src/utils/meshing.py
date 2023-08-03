# --- Built-ins ---

# --- Internal ---
from src.base import WingPropInfo
from RethorstCorrection_pyf90.mod_vlm_mesh import vlm_mesh

# --- External ---
import numpy as np


def meshing(span: float, prop_locations: np.array, prop_radii: np.array, nr_props: int, 
            spanwise_discretisation_wing: int, spanwise_discretisation_propeller: int, total_panels: int):
    y_vlm = np.zeros(total_panels+1, order='F')
    # TODO: Move these calculations to Python, makes debugging easier
    vlm_mesh(span=span,
             y_vlm=y_vlm,
             prop_locations=prop_locations,
             prop_radii=prop_radii,
             nr_props=nr_props,
             panels_wing=spanwise_discretisation_wing,
             panels_jet=spanwise_discretisation_propeller)

    nx = 2  # number of chordwise nodal points (should be odd)
    # number of spanwise nodal points for the outboard segment
    ny = wing_prop.spanwise_panels+1

    assert ny % 2 != 0

    mesh = np.zeros((nx, ny, 3))

    mesh[:, :, 2] = 0.0
    mesh[:, :, 1] = y_vlm
    mesh[:, :, 0] = np.zeros(ny)
    mesh[1, :, 0] = np.ones(ny)

    return mesh
