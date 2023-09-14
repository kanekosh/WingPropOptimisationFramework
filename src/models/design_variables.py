# --- Built-ins ---

# --- Internal ---
from src.base import WingPropInfo

# --- External ---
import openmdao.api as om
import numpy as np

# TODO: move this to a new variable directory
class DesignVariables(om.IndepVarComp):
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo)

    def setup(self):
        # === Options ===
        wingpropinfo = self.options['WingPropInfo']
        
        # === Output ===
        # Propeller Design Variables
        for index, _ in enumerate(wingpropinfo.propeller):
            self.add_output(f"rotor_{index}_twist", val=wingpropinfo.propeller[index].twist)
            self.add_output(f"rotor_{index}_chord", val=wingpropinfo.propeller[index].chord)
            self.add_output(f"rotor_{index}_rot_rate", val=wingpropinfo.propeller[index].rot_rate)
        
        # Wing Design Variables
        self.add_output("span", val=wingpropinfo.wing.span, units="m")
        self.add_output("twist", val=wingpropinfo.wing.twist, units="deg")
        self.add_output("chord", val=np.ones(len(wingpropinfo.wing.chord)), units="m")
        
        # === Aero optimized starting point ===
        # self.add_output("span", val=wingpropinfo.wing.span, units="m")
        # self.add_output("twist", val=[4.56517186, 8.21258289, -2.41785719, 8.21258318, 4.56517122], units="deg")
        # self.add_output("chord", val=[0.62713202, 0.17093205, 3.40830288,  0.17093205, 0.62713202], units="m")