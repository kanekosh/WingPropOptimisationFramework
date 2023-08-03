# --- Built-ins ---

# --- Internal ---
from src.base import WingPropInfo, WingInfo, PropInfo
from src.integration.model_coupling import WingSlipstreamProp

# --- External ---
import numpy as np
import openmdao.api as om

spanwise_discretisation_propeller_BEM = 10

prop1 = PropInfo(label='Prop1',
                 prop_location=-1.,
                 nr_blades=4,
                 rot_rate=500.,
                 chord=np.ones(spanwise_discretisation_propeller_BEM, 
                                order='F'),
                 twist=np.ones(spanwise_discretisation_propeller_BEM, 
                                order='F'),
                 span=np.array(np.linspace(0.1, 1.0, spanwise_discretisation_propeller_BEM), 
                                order='F')
                 )

prop2 = PropInfo(label='Prop1',
                 prop_location=-1.,
                 nr_blades=4,
                 rot_rate=500.,
                 chord=np.ones(spanwise_discretisation_propeller_BEM, 
                                order='F'),
                 twist=np.ones(spanwise_discretisation_propeller_BEM, 
                                order='F'),
                 span=np.array(np.linspace(0.1, 1.0, spanwise_discretisation_propeller_BEM), 
                                order='F')
                 )

propellers = [prop1, prop2]
wingpropinfo = WingPropInfo(nr_props=2,
                            spanwise_discretisation_wing=30,
                            spanwise_discretisation_propeller=5,
                            spanwise_discretisation_propeller_BEM=spanwise_discretisation_propeller_BEM,
                            propeller=propellers
                            )

class WingSlipstreamPropAnalysis(om.Group):
    def initialize(self):
        self.declare.options('WingPropInfo', default=WingPropInfo())
    
    def setup(self):
        self.add_subsystem('PropellerSlipstreamWingModel',
                           subsys=WingSlipstreamPropAnalysis(WingPropInfo=wingpropinfo))
        
    def configure(self):
        # Empty because we do analysis
        ...

prob = om.Problem()
prob.model = WingSlipstreamPropAnalysis(WingPropInfo=wingpropinfo)
prob.setup()
prob.run_model()