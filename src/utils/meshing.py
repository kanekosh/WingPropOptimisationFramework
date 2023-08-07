# --- Built-ins ---

# --- Internal ---
from RethorstCorrection_pyf90.mod_vlm_mesh import vlm_mesh

# --- External ---
import numpy as np


def meshing(span: float, chord: float, prop_locations: np.array, prop_radii: np.array, nr_props: int, 
            spanwise_discretisation_wing: int, spanwise_discretisation_propeller: int, total_nodes: int):
        # TODO: include variable chord
    ny = total_nodes
    y_vlm = np.zeros(ny, order='F')
    # TODO: Move these calculations to Python, makes debugging easier
    vlm_mesh(span=span,
             y_vlm=y_vlm,
             prop_locations=prop_locations,
             prop_radii=np.array(prop_radii[:, ::-1], order='F'), # reverse the radius input
             nr_props=nr_props,
             panels_wing=spanwise_discretisation_wing,
             panels_jet=spanwise_discretisation_propeller)

    nx = 2  # number of chordwise nodal points (should be odd)
    # number of spanwise nodal points for the outboard segment

    mesh = np.zeros((nx, ny, 3), order='F')

    mesh[:, :, 2] = 0.0
    mesh[:, :, 1] = y_vlm
    mesh[:, :, 0] = np.zeros(ny)
    mesh[1, :, 0] = np.ones(ny)*chord

    return np.array(mesh, order='F')
