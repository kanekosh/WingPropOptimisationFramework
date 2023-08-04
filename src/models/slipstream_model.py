# --- Built-ins ---

# --- Internal ---
from src.base import WingPropInfo
from rethorst.openmdao.om_rethorst_velocityinterpolation import RETHORST_velocityinterpolation
from rethorst.openmdao.om_rethorst_correctionmatrix import RETHORST_correction

# --- External ---
import openmdao.api as om

class SlipStreamModel(om.Group):
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo)

    def setup(self):
        # === Options ===
        wingpropinfo = self.options["WingPropInfo"]
        
        # === Components ===
        self.add_subsystem('interpolation',
                    subsys=RETHORST_velocityinterpolation(propeller_quantity=wingpropinfo.nr_props,
                                                propeller_discretisation_BEM=wingpropinfo.spanwise_discretisation_propeller_BEM,
                                                propeller_discretisation=wingpropinfo.spanwise_discretisation_propeller),
                    promotes_inputs=['propeller_radii_BEM',
                                    'propeller_velocity_BEM'],
                    promotes_outputs=['propeller_radii',
                                      'propeller_velocity'])

        self.add_subsystem('correction',
                           subsys=RETHORST_correction(propeller_quantity=wingpropinfo.nr_props,
                                                      propeller_discretisation=wingpropinfo.spanwise_discretisation_propeller,
                                                      wing_mesh_size=wingpropinfo.spanwise_discretisation_nodes),
                           promotes_inputs=['propeller_locations',
                                            'propeller_radii',
                                            'wing_mesh',
                                            'wing_mesh_control_points',
                                            'propeller_velocity'],
                           promotes_outputs=['correction_matrix',
                                             'velocity_distribution'])

        # === Explicit connections ===