# --- Built-ins ---

# --- Internal ---
from src.base import PropInfo
import helix.parameters.simparam_def as py_simparam_def
import helix.references.references_def as py_ref_def
import helix.geometry.geometry_def as py_geo_def
import helix.geometry.geometry_def_parametric as py_geo_def_parametric
import helix.openmdao.om_helix as om_helix

# --- External ---
import openmdao.api as om
import numpy as np


class PropellerModel(om.Group):
    def initialize(self):
        self.options.declare('PropellerInfo', default=PropInfo())
        
    def setup(self):
        self.propellerinfo = self.options["PropellerInfo"]

        # Set Up Helix Case
        simparam_def = self._simparam_definition()
        references_def = self._references_definition()
        geometry_def = self._geometry_definition()

        # Define OpenMDAO Problem
        prob = om.Problem()
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["optimizer"] = "SLSQP"

        prob.model.add_subsystem(
            "om_helix",
            om_helix.HELIX_Group(
                simparam_def=simparam_def,
                references_def=references_def,
                geometry_def=geometry_def,
                thrust_calc=True,
                torque_calc=True,
                moment_calc=True,
                power_calc=True,
                loads_calc=True,
                velocity_distribution_calc=False,
                force_distribution_calc=False,
            ),
        )
    
    # rst simparam
    def _simparam_definition(self):
        simparam = py_simparam_def.t_simparam_def()
        simparam.basename = "exampleOpt"

        simparam.nt = 5
        simparam.t_start = 0.0
        simparam.t_end = 0.1

        simparam.nt_rev = 30

        simparam.v_inf = np.array([0.0, 0.0, -1.0])
        simparam.rho_inf = 1.25

        return simparam


    # rst simparam (end)

    # rst ref
    def _references_definition(self):
        # Initialize Reference Frame Defintions Holder
        references_def = py_ref_def.t_references_def()

        # Hub Frame
        Hub = py_ref_def.t_frame_def()
        Hub.Name = "Hub"
        Hub.Parent = "Root"
        Hub.origin = np.array([0.0, 0.0, 0.0])
        Hub.orientation = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        Hub.moving = False

        # Append to References
        references_def.append_frame(Hub)

        return references_def


    # rst ref (end)

    # rst geodef
    def _geometry_definition(self):
        # Initialize Geometry Component Definitions Holder
        geometry_def = py_geo_def.t_geometry_def()

        # ---------------------------- Blade Parameters -------------------------- #
        rotor1 = py_geo_def_parametric.t_geometry_def_parametric()
        rotor1.CompName = "Rotor1"
        rotor1.CompType = "rotor"
        rotor1.RefName = "Hub"

        # Reference Parameters
        N_span = 1
        rotor1.ref_point = np.array([0.0, 0.0, 0.0])
        rotor1.ref_chord_frac = 0.5

        # Symmetry Parameters
        rotor1.symmetry = False
        rotor1.symmetry_point = np.array([0.0, 0.0, 0.0])
        rotor1.symmetry_normal = np.array([0.0, 1.0, 0.0])

        # Mirror Parameters
        rotor1.mirror = False
        rotor1.mirror_point = np.array([0.0, 0.0, 0.0])
        rotor1.mirror_normal = np.array([0.0, 1.0, 0.0])

        # Initialize Rotor and Allocate Arrays
        rotor1.initialize_parametric_geometry_definition(N_span)

        rotor1.multiple = True
        rotor1.multiplicity = {
            "mult_type": "rotor",
            "n_blades": self.propellerinfo.nr_blades,
            "rot_axis": self.propellerinfo.rotation_axis,
            "rot_rate": self.propellerinfo.rot_rate,
            "psi_0": 0.0,
            "hub_offset": self.propellerinfo.hub_offset,
            "n_dofs": 0,
        }

        # ------------------------ Blade Section Definition ---------------------- #
        # Chord 1 ------------------
        rotor1.sec[0].chord = 0.02
        rotor1.sec[0].twist = 10.0
        rotor1.sec[0].thick = 0.12 * 0.02
        rotor1.sec[0].alpha_0 = 20.6 * np.pi / 180.0
        rotor1.sec[0].alpha_L0 = -2.0 * np.pi / 180.0
        rotor1.sec[0].Cl_alpha = 1.7 * np.pi
        rotor1.sec[0].M = 50.0

        # Span 1  ------------------
        rotor1.span[0].span = 0.1  # 0.05
        rotor1.span[0].sweep = 0.0
        rotor1.span[0].dihed = 0.0
        rotor1.span[0].N_elem_span = 10
        rotor1.span[0].span_type = "uniform"

        # Chord 2 ------------------
        rotor1.sec[1].chord = 0.02
        rotor1.sec[1].twist = 10.0
        rotor1.sec[1].thick = 0.12 * 0.02
        rotor1.sec[1].alpha_0 = 20.6 * np.pi / 180.0
        rotor1.sec[1].alpha_L0 = -2.0 * np.pi / 180.0
        rotor1.sec[1].Cl_alpha = 1.7 * np.pi
        rotor1.sec[1].M = 50.0

        # Append To Vehicle
        geometry_def.append_component(rotor1)

        return geometry_def