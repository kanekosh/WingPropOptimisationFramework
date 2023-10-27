# --- Built-ins ---

# --- Internal ---

# --- External ---
import numpy as np


def meshing(span: float, chord: float, prop_locations: np.array, prop_radii: np.array, nr_props: int, 
            spanwise_discretisation_wing: int, spanwise_panels_propeller: int):
    # This function currently assumes that no wing-tip propellers are configured!
    y_vlm = np.array([-span/2], order='F')
    spanwise_nodes_propeller = spanwise_panels_propeller+1

    nr_wing_regions = nr_props+1
    wing_panels_regional = int(spanwise_discretisation_wing/nr_wing_regions)
    
    # Check whether ny is odd
    ny = wing_panels_regional*(nr_props+1)+spanwise_nodes_propeller*nr_props

    if ny%2==0:
        wing_panels_regional+=1
    
    # Update ny
    ny = wing_panels_regional*(nr_props+1)+spanwise_nodes_propeller*nr_props    
    assert(ny%2==1), 'ny should be odd number'
    
    for iprop in range(nr_props):
        start = y_vlm[-1]
        iprop_loc = prop_locations[iprop]
        prop_left = iprop_loc-max(prop_radii[iprop])
        prop_right = iprop_loc+max(prop_radii[iprop])
        
        new_mesh = np.linspace(start, prop_left, wing_panels_regional)
        new_mesh = np.concatenate((new_mesh[1:-1],
                                  np.linspace(prop_left, prop_right, spanwise_nodes_propeller))
                                  )
        
        y_vlm = np.concatenate((y_vlm, new_mesh))

    new_mesh = np.linspace(prop_right, span/2, wing_panels_regional)
    y_vlm = np.concatenate((y_vlm, new_mesh[1:]))
    
    nx = 2  # number of chordwise nodal points (should be odd)
    ny = len(y_vlm)
    # number of spanwise nodal points for the outboard segment

    mesh = np.zeros((nx, ny, 3), order='F')

    mesh[:, :, 2] = 0.0
    mesh[:, :, 1] = y_vlm
    mesh[:, :, 0] = np.zeros(ny)
    mesh[0, :, 0] = np.zeros(ny)
    mesh[1, :, 0] = np.ones(ny)*chord

    return np.array(mesh, order='F')
