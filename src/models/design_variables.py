# --- Built-ins ---

# --- Internal ---
from src.base import WingPropInfo

# --- External ---
import openmdao.api as om
import numpy as np


class DesignVariables(om.IndepVarComp):
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo)

    def setup(self):
        # === Options ===
        wingpropinfo = self.options['WingPropInfo']
        
        # === Output ===
        # Propeller Design Variables
        for index, _ in enumerate(wingpropinfo.propeller):
            self.add_output(f"rotor_{index}_twist", val=wingpropinfo.propeller[index].span, units="deg")
            self.add_output(f"rotor_{index}_chord", val=wingpropinfo.propeller[index].span, units="m")
            self.add_output(f"rotor_{index}_rot_rate", val=wingpropinfo.propeller[index].span, units="rad/s")
        
        # Wing Design Variables
        self.add_output("span", val=wingpropinfo.wing.span, units="m")
        self.add_output("twist", val=wingpropinfo.wing.twist, units="deg")
        self.add_output("chord", val=wingpropinfo.wing.chord, units="m")