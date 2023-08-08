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

wing_twist = 0.
wing_chord = 0.24

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

prop2 = PropInfo(label='Prop1',
                 prop_location=0.332,
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

parameters = ParamInfo(vinf=40.,
                       wing_aoa=2.,
                       mach_number=0.2,
                       reynolds_number=5.e6,
                       speed_of_sound=333.4)


wing = WingInfo(label='SampleWing',
                span=0.748*2*0.976,
                chord=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*wing_chord,
                twist=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*wing_twist,
                empty_weight=0.,
                CL0 = 0.283
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
    # === Generate numerical data ===
    CL_numerical = []
    J = np.array([0.796, 1.0])
    rot_rate = (wingpropinfo.parameters.vinf/(J*2.*prop_radius)) * 2.*np.pi # in rad/s
    angles = np.arange(-8, 10, 1)
    
    prob = om.Problem()

    for index_rotational, _ in enumerate(rot_rate):
        CL_numerical_tmp = []
        for index_propeller, _ in enumerate(wingpropinfo.propeller):
            wingpropinfo.propeller[index_propeller].rot_rate = rot_rate[index_rotational]
        
        for angle in angles:
            print(f'Angle of attack: {angle: ^10}')
            wingpropinfo.parameters.wing_aoa = angle
            
            prob.model = WingSlipstreamPropAnalysis(WingPropInfo=wingpropinfo)
            prob.setup()
            prob.run_model()
            
            CL_numerical_tmp.append(prob["PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.aero_funcs.CL"].tolist()[0])
            
        CL_numerical.append(CL_numerical_tmp)

    # === Load in (experimental) validation data ===
    validation_file = os.path.join(BASE_DIR, 'data', 'PROWIM_validation_conventional.txt')
    validation_data = pd.read_csv(validation_file, delimiter=',', skiprows=22)
    
    # Validation data for J=inf (prop-off)
    n=0
    index1 = n*19
    index2 = (n+1)*19
    aoa = data['AoA'][index1:index2]
    CL_Jinf = data['CL'][index1:index2]
    CD_Jinf = data['CD'][index1:index2]
    J_inf = data['J'][index1+1]
    
    # Validation data for J=1
    n=1
    index1 = n*19
    index2 = (n+1)*19
    aoa = validation_data['AoA'][index1:index2]
    CL_J1 = validation_data['CL'][index1:index2]
    CD_J1 = validation_data['CD'][index1:index2]
    J_1 = validation_data['J'][index1+1]
    
    # Validation data for J=0.796
    n=4
    index1 = n*19
    index2 = (n+1)*19
    aoa = validation_data['AoA'][index1:index2]
    CL_J0796 = validation_data['CL'][index1:index2]
    CD_J0796 = validation_data['CD'][index1:index2]
    J_0796 = validation_data['J'][index1+1]
    
    # === Plot results ===
    plt.style.use(niceplots.get_style())
    _, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(angles, CL_numerical[0], label='Numerical, J=0.796', color='b')
    ax.plot(angles, CL_numerical[1], label=f'Numerical, J=1.0', color='orange')
    ax.scatter(aoa, CL_J0796, label=f'Experimental, J=0.7962', color='b')
    ax.scatter(aoa, CL_J1, label=f'Experimental, J=1.0', color='orange')
    
    ax.set_xlabel("Angle of Attack (deg)")
    ax.set_ylabel(r"Lift Coefficient ($C_L$)")
    ax.legend()

    niceplots.adjust_spines(ax, outward=True)

    plt.savefig(os.path.join(BASE_DIR, 'figures', 'PROWIM_VALIDATION.png'))