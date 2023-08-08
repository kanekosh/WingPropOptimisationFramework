# --- Built-ins ---
import os
from pathlib import Path
import json

# --- Internal ---
from src.base import ParamInfo, WingPropInfo, WingInfo, PropInfo, AirfoilInfo
from src.integration.model_coupling import WingSlipstreamProp

# --- External ---
import numpy as np
import pandas as pd
import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots


BASE_DIR = Path(__file__).parents[0]


# === Read in PROWIM data ===
with open(os.path.join(BASE_DIR, 'data', 'PROWIM.json'), 'r') as file:
    data = json.load(file)

prop_radius = 0.1185
ref_point = data['ref_point']
span = data['span']
twist = data['twist']
chord = data['chord']

alpha_0 = data['alpha_0']
alpha_L0 = data['alpha_L0']
Cl_alpha = data['Cl_alpha']
M = data['M']

spanwise_discretisation_propeller_BEM = len(span)


prop1 = PropInfo(label='Prop1',
                 prop_location=-0.332,
                 nr_blades=4,
                 rot_rate=300.,
                 chord=np.array(chord, order='F'),
                 twist=np.array(twist, order='F'),
                 span=np.array(span, order='F'),
                 airfoils=[AirfoilInfo(label=f'Foil_{index}',
                                       Cl_alpha=Cl_alpha[index],
                                       alpha_L0=alpha_L0[index],
                                       alpha_0=alpha_0[index],
                                        M=M[index])
                           for index in range(spanwise_discretisation_propeller_BEM+1)],
                 ref_point=ref_point
                 )

prop2 = prop1

parameters = ParamInfo(vinf=40.,
                       wing_aoa=2.,
                       mach_number=0.2,
                       reynolds_number=5.e6,
                       speed_of_sound=333.4)


wing = WingInfo(label='SampleWing',
                span=0.748*2*0.976,
                chord=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*0.24,
                twist=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*0.1,
                empty_weight=10.
                )


wingpropinfo = WingPropInfo(nr_props=2,
                            spanwise_discretisation_wing=60,
                            spanwise_discretisation_propeller=11,
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
    J = np.array([0.796, 1.0])
    rot_rate = (wingpropinfo.parameters.vinf/J) * 2*np.pi # in rad/s
    
    prob = om.Problem()
    
    for index_rotational, _ in enumerate(rot_rate):
        for index_propeller, _ in enumerate(wingpropinfo.propeller):
            wingpropinfo.propeller[index_propeller].rot_rate = rot_rate[index_rotational]
        
        prob.model = WingSlipstreamPropAnalysis(WingPropInfo=wingpropinfo)
        prob.setup()
        prob.run_model()
        
        Cl = prob["PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.aero_funcs.liftcoeff.Cl"]
