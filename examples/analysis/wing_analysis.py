# --- Built-ins ---
import os
from pathlib import Path

# --- Internal ---
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.utils.constants import grav_constant

# --- External ---
import numpy as np
import openmdao.api as om
import pandas as pd
import niceplots
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parents[0]

def lift_calculator(alpha: float)->list[float]:
    # Create a dictionary to store options about the surface
    y_panels = 21
    x_panels = 9
    vinf = 40.
    chord = 0.24
    span = 0.73*2*0.952
    mesh_dict = {"num_y": y_panels, "num_x": x_panels, "wing_type": "rect", 
                 "symmetry": False, "num_twist_cp": int(y_panels/2),
                 "span": span, "root_chord": 1.}

    mesh = generate_mesh(mesh_dict)
    
    chord_cp = np.ones(10)*chord
    twist_cp = np.zeros(10)

    surface = {
        # === Wing definition ===
        "name": "wing",  # name of the surface
        "symmetry": False,  # if true, model one half of wing reflected across the plane y = 0
        "S_ref_type": "wetted",  # how we compute the wing area,        # can be 'wetted' or 'projected'
        "fem_model_type": "tube",
        "thickness_cp": np.array([0.1, 0.2, 0.3]),
        "twist_cp": twist_cp,
        "mesh": mesh,
        "chord_cp": chord_cp,
        "CL0": 0.283,  # CL of the surface at alpha=0
        "CD0": 0.015,  # CD of the surface at alpha=0
        "k_lam": 0.05,  # percentage of chord with laminar flow, used for viscous drag
        "t_over_c_cp": np.array([0.15]),  # thickness over chord ratio (NACA0015)
        "c_max_t": 0.303,  # chordwise location of maximum (NACA0015) thickness
        "with_viscous": True,
        "with_wave": False,  # if true, compute wave drag
        # Structural values are based on aluminum 7075
        "E": 70.0e9,  # [Pa] Young's modulus of the spar
        "G": 30.0e9,  # [Pa] shear modulus of the spar
        "yield": 500.0e6 / 2.5,  # [Pa] yield stress divided by 2.5 for limiting case
        "mrho": 3.0e3,  # [kg/m^3] material density
        "fem_origin": 0.35,  # normalized chordwise location of the spar
        "wing_weight_ratio": 2.0,
        "struct_weight_relief": False,  # True to add the weight of the structure to the loads on the structure
        "distributed_fuel_weight": False,
        # Constraints
        "exact_failure_constraint": False,  # if false, use KS function
    }

    # Create the problem and assign the model group
    prob = om.Problem()

    # Add problem information as an independent variables component
    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output("v", val=vinf, units="m/s")
    indep_var_comp.add_output("alpha", val=alpha, units="deg")
    indep_var_comp.add_output("Mach_number", val=0.84)
    indep_var_comp.add_output("re", val=1.0e6, units="1/m")
    indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
    indep_var_comp.add_output("CT", val=grav_constant * 17.0e-6, units="1/s")
    indep_var_comp.add_output("R", val=11.165e6, units="m")
    indep_var_comp.add_output("W0", val=0.4 * 3e5, units="kg")
    indep_var_comp.add_output("speed_of_sound", val=295.4, units="m/s")
    indep_var_comp.add_output("load_factor", val=1.0)
    indep_var_comp.add_output("empty_cg", val=np.zeros((3)), units="m")
    
    indep_var_comp.add_output("velocity_distribution", val=vinf*np.ones(y_panels-1), units="m/s")
    indep_var_comp.add_output("rethorst_correction", val=np.zeros(((y_panels-1)*(x_panels-1),
                                                                       (y_panels-1)*(x_panels-1)))
                              )

    prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

    aerostruct_group = AerostructGeometry(surface=surface)

    name = "wing"

    # Add tmp_group to the problem with the name of the surface.
    prob.model.add_subsystem(name, aerostruct_group)

    point_name = "AS_point_0"

    # Create the aero point group and add it to the model
    AS_point = AerostructPoint(surfaces=[surface])

    prob.model.add_subsystem(
        point_name,
        AS_point,
        promotes_inputs=[
            "v",
            "alpha",
            "Mach_number",
            "re",
            "rho",
            "CT",
            "R",
            "W0",
            "speed_of_sound",
            "empty_cg",
            "load_factor",
        ],
    )
    
    prob.model.connect("velocity_distribution", "AS_point_0.coupled.aero_states.velocity_distribution")
    prob.model.connect("rethorst_correction", "AS_point_0.coupled.aero_states.rethorst_correction")

    com_name = point_name + "." + name + "_perf"
    prob.model.connect(name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed")
    prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")

    # Connect aerodyamic mesh to coupled group mesh
    prob.model.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")

    # Connect performance calculation variables
    prob.model.connect(name + ".radius", com_name + ".radius")
    prob.model.connect(name + ".thickness", com_name + ".thickness")
    prob.model.connect(name + ".nodes", com_name + ".nodes")
    prob.model.connect(name + ".cg_location", point_name + "." + "total_perf." + name + "_cg_location")
    prob.model.connect(name + ".structural_mass", point_name + "." + "total_perf." + name + "_structural_mass")
    prob.model.connect(name + ".t_over_c", com_name + ".t_over_c")

    # Set up the problem
    prob.setup()
    prob.run_model()
    
    if alpha!=0.:
        return prob["AS_point_0.wing_perf.aero_funcs.CL"]
    else:
        return prob["AS_point_0.wing_perf.aero_funcs.CL"]
    
validation_file = os.path.join(BASE_DIR, 'data', 'PROWIM_validation_conventional.txt')
validation_data = pd.read_csv(validation_file, delimiter=',', skiprows=22)

# Validation data for J=inf (prop-off)
n=0
index1 = n*19
index2 = (n+1)*19
aoa = validation_data['AoA'][index1:index2]
CL_Jinf = validation_data['CL'][index1:index2]
CD_Jinf = validation_data['CD'][index1:index2]
J_inf = validation_data['J'][index1+1]

angles = aoa #np.linspace(-8., 10., 10)
CL = []

for angle in angles:
    print(f'Angle of attack: {angle: ^10}')
    CL.append(lift_calculator(alpha=angle))

# === Plot results ===
plt.style.use(niceplots.get_style())
_, ax = plt.subplots(figsize=(10, 7))

ax.plot(angles, CL, label=f'Numerical, J=inf', color='grey')

ax.scatter(aoa, CL_Jinf, label=f'Experimental, J=inf', color='grey')


ax.set_xlabel("Angle of Attack (deg)")
ax.set_ylabel(r"Lift Coefficient ($C_L$)")
ax.legend(fontsize='12')

niceplots.adjust_spines(ax, outward=True)

plt.savefig(os.path.join(BASE_DIR, 'figures', 'PROWIM_WING_VALIDATION.png'))