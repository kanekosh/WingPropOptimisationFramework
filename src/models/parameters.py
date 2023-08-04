# --- Built-ins ---

# --- Internal ---
from src.base import WingPropInfo

# --- External ---
import openmdao.api as om
import numpy as np


class Parameters(om.IndepVarComp):
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo)

    def setup(self):
        # === Options ===
        wingpropinfo = self.options['WingPropInfo']

        # === Outputs ===
        # Freestream Parameters
        self.add_output("vinf", val=wingpropinfo.parameters.vinf, units="m/s")
        self.add_output("alpha", val=wingpropinfo.parameters.wing_aoa, units="deg")
        self.add_output("Mach_number", val=wingpropinfo.parameters.mach_number)
        self.add_output(
            "re", val=wingpropinfo.parameters.reynolds_number, units="1/m")
        self.add_output(
            "rho", val=wingpropinfo.parameters.air_density, units="kg/m**3")
        self.add_output("CT", val=wingpropinfo.parameters.CT, units="1/s")
        self.add_output("R", val=wingpropinfo.parameters.R, units="m")
        self.add_output("W0", val=wingpropinfo.wing.empty_weight, units="kg")
        self.add_output("speed_of_sound",
                        val=wingpropinfo.parameters.speed_of_sound, units="m/s")
        self.add_output("load_factor", val=wingpropinfo.wing.load_factor)
        self.add_output("empty_cg", val=wingpropinfo.wing.empty_cg, units="m")
        
        # Propeller Parameters
        for index, _ in enumerate(wingpropinfo.propeller):
            self.add_output(f"rotor_{index}_radius", val=wingpropinfo.propeller[index].span, units="m")
        
        self.add_output("propeller_locations",
                        val=wingpropinfo.prop_locations, units="m")
        self.add_output("propeller_radii",
                        val=wingpropinfo.prop_radii, units="m")
        self.add_output("wing_mesh", val=wingpropinfo.vlm_mesh, units="m")
        self.add_output("wing_mesh_control_points",
                        val=wingpropinfo.vlm_mesh_control_points, units="m")