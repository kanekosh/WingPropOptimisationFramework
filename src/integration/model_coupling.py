# --- Built-ins ---
import unittest

# --- Internal ---
from src.base import WingPropInfo, WingInfo, PropInfo
from src.models.propeller_model import PropellerModel
from src.models.wing_model import WingModel
from src.models.slipstream_model import SlipStreamModel
from src.models.parameters import Parameters
from src.models.design_variables import DesignVariables

# --- External ---
import numpy as np
import openmdao.api as om


class WingSlipstreamProp(om.Group):
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo())

    def setup(self):
        # === Options ===
        wingpropinfo = self.options["WingPropInfo"]

        # === Components ===
        self.add_subsystem('PARAMETERS', subsys=Parameters(
            WingPropInfo=wingpropinfo))

        self.add_subsystem('DESIGNVARIABLES', subsys=DesignVariables(
            WingPropInfo=wingpropinfo))

        for propeller_nr in range(wingpropinfo.nr_props):
            self.add_subsystem(f'HELIX_{propeller_nr}',
                               subsys=PropellerModel(PropellerInfo=wingpropinfo.propeller[propeller_nr]))

        self.add_subsystem('RETHORST',
                           subsys=SlipStreamModel(WingPropInfo=wingpropinfo))

        self.add_subsystem('OPENAEROSTRUCT',
                           subsys=WingModel(WingPropInfo=wingpropinfo))

        # === Explicit connections ===
        # PARAMS to HELIX
        for index, _ in enumerate(wingpropinfo.propeller):
            self.connect(f"PARAMETERS.rotor_{index}_radius",
                         f"HELIX_{index}.geodef_parametric_0_span")

        # PARAMS to RETHORST
        self.connect("PARAMETERS.vinf",
                     "RETHORST.interpolation.vinf")

        self.connect("PARAMETERS.propeller_locations",
                     "RETHORST.correction.propeller_locations")
        self.connect("PARAMETERS.vinf",
                     "RETHORST.correction.vinf")
        self.connect("PARAMETERS.wing_mesh",
                     "RETHORST.correction.wing_mesh")
        self.connect("PARAMETERS.wing_mesh_control_points",
                     "RETHORST.correction.wing_mesh_control_points")

        # PARAMS to OPENAEROSTRUCT

        # DVs to HELIX
        for index, _ in enumerate(wingpropinfo.propeller):
            self.connect(f"DESIGNVARIABLES.rotor_{index}_twist",
                         f"HELIX_{index}.geodef_parametric_0_twist")
            self.connect(f"DESIGNVARIABLES.rotor_{index}_chord",
                         f"HELIX_{index}.geodef_parametric_0_chord")
            self.connect(f"DESIGNVARIABLES.rotor_{index}_rot_rate",
                         f"HELIX_{index}.geodef_parametric_0_rot_rate")
        
        # DVs to OPENAEROSTRUCT
        
        
        # HELIX to RETHORST
        self.connect("HELIX.helix.rotorcomp_0_radii",
                     "RETHORST.interpolation.propeller_radii_BEM")
        self.connect("HELIX.helix.rotorcomp_0_velocity_distribution",
                     "RETHORST.interpolation.propeller_velocity_BEM")

        # RETHORST to OPENAEROSTRUCT
        self.connect("RETHORST.velocity_distribution",
                     "OPENAEROSTRUCT.AS_point_0.coupled.aero_states.velocity_distribution")
        self.connect("RETHORST.rethorst_correction",
                     "OPENAEROSTRUCT.AS_point_0.coupled.aero_states.rethorst_correction")
