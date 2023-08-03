# --- Built-ins ---

# --- Internal ---
from src.base import WingPropInfo, WingInfo
from src.utils.meshing import meshing
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.utils.constants import grav_constant

# --- External ---
import numpy as np
import openmdao.api as om

class SimpleDualProp(om.group):
    def setup(self):
        # Add problem information as an independent variables component
        indep_var_comp = om.IndepVarComp()
        indep_var_comp.add_output("v", val=100., units="m/s")
        indep_var_comp.add_output("alpha", val=5.0, units="deg")
        indep_var_comp.add_output("Mach_number", val=0.84)
        indep_var_comp.add_output("re", val=1.0e6, units="1/m")
        indep_var_comp.add_output("rho", val=0.38, units="kg/m**3")
        indep_var_comp.add_output("CT", val=grav_constant * 17.0e-6, units="1/s")
        indep_var_comp.add_output("R", val=11.165e6, units="m")
        indep_var_comp.add_output("W0", val=0.4 * 3e5, units="kg")
        indep_var_comp.add_output("speed_of_sound", val=295.4, units="m/s")
        indep_var_comp.add_output("load_factor", val=1.0)
        indep_var_comp.add_output("empty_cg", val=np.zeros((3)), units="m")

        indep_var_comp.add_output("velocity_distribution", val=velocity_distribution, units="m/s")
        indep_var_comp.add_output("rethorst_correction", val=correction) #np.zeros( ((y_panels-1), (y_panels-1)) ) )