# --- Built-ins ---
import unittest

# --- Internal ---
from src.base import WingPropInfo
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
        self.options.declare('WingPropInfo', default=WingPropInfo)

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
                               subsys=PropellerModel(ParamInfo=wingpropinfo.parameters,
                                                     PropInfo=wingpropinfo.propeller[propeller_nr]))

        self.add_subsystem('RETHORST',
                           subsys=SlipStreamModel(WingPropInfo=wingpropinfo))

        self.add_subsystem('OPENAEROSTRUCT',
                           subsys=WingModel(WingPropInfo=wingpropinfo))

        # === Explicit connections ===
        # PARAMS to HELIX
        for index, _ in enumerate(wingpropinfo.propeller):
            self.connect(f"PARAMETERS.rotor_{index}_radius",
                         f"HELIX_{index}.om_helix.geodef_parametric_0_span")

        # PARAMS to RETHORST
        self.connect("PARAMETERS.vinf",
                     "RETHORST.interpolation.vinf")

        self.connect("PARAMETERS.propeller_locations",
                     "RETHORST.propeller_locations")
        self.connect("PARAMETERS.vinf",
                     "RETHORST.correction.vinf")
        self.connect("PARAMETERS.wing_mesh",
                     "RETHORST.wing_mesh")
        self.connect("PARAMETERS.wing_mesh_control_points",
                     "RETHORST.wing_mesh_control_points")

        # PARAMS to OPENAEROSTRUCT
        self.connect('PARAMETERS.vinf',
                     'OPENAEROSTRUCT.v')
        self.connect('PARAMETERS.alpha',
                     'OPENAEROSTRUCT.alpha')
        self.connect('PARAMETERS.Mach_number',
                     'OPENAEROSTRUCT.Mach_number')
        self.connect('PARAMETERS.re',
                     'OPENAEROSTRUCT.re')
        self.connect('PARAMETERS.rho',
                     'OPENAEROSTRUCT.rho')
        self.connect('PARAMETERS.CT',
                     'OPENAEROSTRUCT.CT')
        self.connect('PARAMETERS.R',
                     'OPENAEROSTRUCT.R')
        self.connect('PARAMETERS.W0',
                     'OPENAEROSTRUCT.W0')
        self.connect('PARAMETERS.speed_of_sound',
                     'OPENAEROSTRUCT.speed_of_sound')
        self.connect('PARAMETERS.load_factor',
                     'OPENAEROSTRUCT.load_factor')
        self.connect('PARAMETERS.empty_cg',
                     'OPENAEROSTRUCT.empty_cg')

        # DVs to HELIX
        for index, _ in enumerate(wingpropinfo.propeller):
            self.connect(f"DESIGNVARIABLES.rotor_{index}_twist",
                         f"HELIX_{index}.om_helix.geodef_parametric_0_twist")
            self.connect(f"DESIGNVARIABLES.rotor_{index}_chord",
                         f"HELIX_{index}.om_helix.geodef_parametric_0_chord")
            self.connect(f"DESIGNVARIABLES.rotor_{index}_rot_rate",
                         f"HELIX_{index}.om_helix.geodef_parametric_0_rot_rate")

        # DVs to OPENAEROSTRUCT
        self.connect('DESIGNVARIABLES.twist',
                     'OPENAEROSTRUCT.wing.twist_cp')
        # self.connect('DESIGNVARIABLES.chord',
        #              'OPENAEROSTRUCT.wing.geometry.chord_cp')
        self.connect('DESIGNVARIABLES.span',
                     'OPENAEROSTRUCT.wing.geometry.span')

        # HELIX to RETHORST
        for index in range(wingpropinfo.nr_props):
            self.connect(f"HELIX_{index}.om_helix.rotorcomp_0_radii",
                        f"RETHORST.interpolation.propeller_radii_BEM_rotor{index}")
            self.connect(f"HELIX_{index}.om_helix.rotorcomp_0_velocity_distribution",
                        f"RETHORST.interpolation.propeller_velocity_BEM_rotor{index}")

        # RETHORST to OPENAEROSTRUCT
        self.connect("RETHORST.velocity_distribution",
                     "OPENAEROSTRUCT.AS_point_0.coupled.aero_states.velocity_distribution")
        self.connect("RETHORST.correction_matrix",
                     "OPENAEROSTRUCT.AS_point_0.coupled.aero_states.rethorst_correction")
