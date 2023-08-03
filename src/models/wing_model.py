# --- Built-ins ---

# --- Internal ---
from src.base import WingPropInfo, WingInfo
from src.utils.meshing import meshing
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

# --- External ---
import numpy as np
import openmdao.api as om


class WingModel(om.Group):
    
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo())
        
    def setup(self):
        # === Options ===
        wingpropinfo = self.options['WingPropInfo']
        winginfo = wingpropinfo.wing
        
        # === Components ===
        twist_cp = winginfo.twist
        
        mesh = self.wingpropinfo.vlm_mesh

        surface = {
                        # === Wing definition ===
                        "name": "wing",  # name of the surface
                        "symmetry": False,  # if true, model one half of wing reflected across the plane y = 0
                        "S_ref_type": "wetted",  # how we compute the wing area,        # can be 'wetted' or 'projected'
                        "fem_model_type": "tube",
                        "thickness_cp": np.array([0.1, 0.2, 0.3]),
                        "twist_cp": twist_cp,
                        "mesh": mesh,
                        "span": winginfo.span,
                        "CL0": 0.0,  # CL of the surface at alpha=0
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

        aerostruct_group = AerostructGeometry(surface=surface)

        name = "wing"

        # Add tmp_group to the problem with the name of the surface.
        self.add_subsystem(name, aerostruct_group)

        point_name = "AS_point_0"

        # Create the aero point group and add it to the model
        AS_point = AerostructPoint(surfaces=[surface])

        self.add_subsystem(
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
        
        # === Explicit connections ===
        com_name = point_name + "." + name + "_perf"
        self.connect(name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed")
        self.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")

        # Connect aerodyamic mesh to coupled group mesh
        self.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")

        # Connect performance calculation variables
        self.connect(name + ".radius", com_name + ".radius")
        self.connect(name + ".thickness", com_name + ".thickness")
        self.connect(name + ".nodes", com_name + ".nodes")
        self.connect(name + ".cg_location", point_name + "." + "total_perf." + name + "_cg_location")
        self.connect(name + ".structural_mass", point_name + "." + "total_perf." + name + "_structural_mass")
        self.connect(name + ".t_over_c", com_name + ".t_over_c")