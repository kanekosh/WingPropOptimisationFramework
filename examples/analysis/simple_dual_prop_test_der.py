# --- Built-ins ---

# --- Internal ---
from src.base import ParamInfo, WingPropInfo, WingInfo, PropInfo, AirfoilInfo
from src.integration.coupled_groups_analysis import WingSlipstreamProp

# --- External ---
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

spanwise_discretisation_propeller_BEM = 9

parameters = ParamInfo(vinf=10.,
                       wing_aoa=2.,
                       mach_number=0.2,
                       reynolds_number=1e6,
                       speed_of_sound=333.4)

wing = WingInfo(label='SampleWing',
                span=10.,
                chord=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F'),
                twist=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*0.1,
                empty_weight=10.
                )

prop1 = PropInfo(label='Prop1',
                 prop_location=-2.,
                 nr_blades=4,
                 rot_rate=300.,
                 chord=np.ones(spanwise_discretisation_propeller_BEM+1,
                               order='F')*0.05,
                 twist=np.array(np.linspace(48, 22, spanwise_discretisation_propeller_BEM+1),
                                order='F'),
                 span=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*0.05,
                 airfoils=[AirfoilInfo(label='SampleFoil',
                                       Cl_alpha=6.22,
                                       alpha_L0=0.,
                                       alpha_0=0.24)]*(spanwise_discretisation_propeller_BEM+1),
                 )

prop2 = PropInfo(label='Prop2',
                 prop_location=2.,
                 nr_blades=4,
                 rot_rate=300.,
                 chord=np.ones(spanwise_discretisation_propeller_BEM+1,
                               order='F')*0.05,
                 twist=np.array(np.linspace(48, 22, spanwise_discretisation_propeller_BEM+1),
                                order='F'),
                 span=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*0.05,
                 airfoils=[AirfoilInfo(label='SampleFoil',
                                       Cl_alpha=6.22,
                                       alpha_L0=0.,
                                       alpha_0=0.24)]*(spanwise_discretisation_propeller_BEM+1),
                 )

wingpropinfo = WingPropInfo(nr_props=2,
                            spanwise_discretisation_wing=30,
                            spanwise_discretisation_propeller=5,
                            spanwise_discretisation_propeller_BEM=spanwise_discretisation_propeller_BEM,
                            propeller=[prop1, prop2],
                            wing=wing,
                            parameters=parameters
                            )


class WingSlipstreamPropAnalysis(om.Group):
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo)

    def setup(self):
        self.add_subsystem('PropellerSlipstreamWingModel',
                           subsys=WingSlipstreamProp(WingPropInfo=wingpropinfo))

    def configure(self):
        # Empty because we do analysis
        ...


if __name__ == '__main__':
    prob = om.Problem()
    prob.model = WingSlipstreamPropAnalysis(WingPropInfo=wingpropinfo)

    prob.setup()
    prob.run_model()
    partials = prob.check_partials(compact_print=True, show_only_incorrect=True, 
                                   includes=['*PropellerSlipstreamWingModel.RETHORST*'], 
                                   form='central', step=1e-10)
    a=1
