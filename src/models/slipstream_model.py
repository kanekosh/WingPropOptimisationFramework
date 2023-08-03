# --- Built-ins ---

# --- Internal ---
from rethorst.openmdao.om_rethorst_velocityinterpolation import RETHORST_velocityinterpolation
from rethorst.openmdao.om_rethorst_correctionmatrix import RETHORST_correction

# --- External ---
import openmdao.api as om

class SlipStreamModel(om.Group):
    def setup(self):
        self.add_subsystem('rethorst',
                    subsys=RETHORST_velocityinterpolation(propeller_quantity=self.wing.nr_props,
                                                propeller_discretisation_BEM=self.wing.spanwise_discretisation_propeller_BEM,
                                                propeller_discretisation=self.wing.spanwise_discretisation_propeller),
                    promotes_inputs=['propeller_radii_BEM',
                                    'propeller_velocity_BEM',
                                    'vinf'],
                    promotes_outputs=['propeller_radii',
                                      'propeller_velocity'])

        self.add_subsystem('rethorst',
                           subsys=RETHORST_correction(propeller_quantity=self.wing.nr_props,
                                                      propeller_discretisation=self.wing.spanwise_discretisation_propeller,
                                                      wing_mesh_size=self.wing.spanwise_panels_total_true+1),
                           promotes_inputs=['propeller_locations',
                                            'propeller_radii',
                                            'wing_mesh',
                                            'wing_mesh_control_points',
                                            'vinf',
                                            'propeller_velocity'],
                           promotes_outputs=['correction_matrix',
                                             'velocity_distribution'])
