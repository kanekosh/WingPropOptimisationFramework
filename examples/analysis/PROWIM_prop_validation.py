# --- Built-ins ---
import os
from pathlib import Path
import json

# --- Internal ---
from src.base import ParamInfo, WingPropInfo, WingInfo, PropInfo, AirfoilInfo
from src.postprocessing.utils import get_niceColors
from src.integration.coupled_groups_analysis import PropAnalysis
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

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

# === Set wing parameters ===
wing_twist = 0.
wing_chord = 0.24
wingspan = 0.73*2.*0.952

# === Set environment variables ===
air_density = 1.2087

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
                 ref_point=np.array([0.0, 0.0, 0.0]), # 0.0174195
                 rotation_axis=np.array([-1., 0., 0.]),
                 hub_orientation=np.array([[1.0, 0.0, 0.0],
                                           [0.0, 1.0*np.cos(np.deg2rad(-0.2)),
                                            1.0*np.sin(np.deg2rad(-0.2))],
                                           [0.0, -1.0*np.sin(np.deg2rad(-0.2)), 1.0*np.cos(np.deg2rad(-0.2))]]),
                 local_refinement=3
                 )

parameters = ParamInfo(vinf=39.19,
                       wing_aoa=0.,
                       mach_number=0.2,
                       reynolds_number=640_000,
                       speed_of_sound=333.4,
                       air_density=air_density)


wing = WingInfo(label='PROWIM_wing',
                span=wingspan,
                thickness=np.ones(10)*0.001, # doesn't matter for this simulation
                chord=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*wing_chord,
                twist=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*wing_twist,
                empty_weight=0.,
                CL0=0.283
                )


wingpropinfo = WingPropInfo(spanwise_discretisation_wing=60,
                            spanwise_discretisation_propeller=31,
                            spanwise_discretisation_propeller_BEM=spanwise_discretisation_propeller_BEM,
                            propeller=[prop1],
                            wing=wing,
                            parameters=parameters
                            )

for index in range(len(PROWIM_wingpropinfo.propeller)):
    PROWIM_wingpropinfo.propeller[index].rot_rate = 644.82864419
    PROWIM_wingpropinfo.propeller[index].twist = np.array([67.79378385, 71.83797648, 61.32955902, 62.9787903,  56.87134101, 58.16629045,
                                                            56.66413092, 54.56196904, 52.76508122, 50.14207845, 48.77576388, 45.81754819,
                                                            44.61299923, 42.01886426, 40.93763764, 38.52984867, 37.65342321, 35.1964771,
                                                            33.97829724, 30.47284116],
                                                                order='F'
                                                        )
    

class PropAnalysis(om.Group):
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo)

    def setup(self):
        self.add_subsystem('PropellerModel',
                           subsys=PropAnalysis(WingPropInfo=PROWIM_wingpropinfo)
                           )


if __name__ == '__main__':
    # === Load in (experimental) validation data ===
    path = os.path.join(BASE_DIR, 'data', 'PROWIM_proponly_data.txt')

    file = pd.read_csv(path, header=22,
                       usecols=['Polar', 'Run', 'AoA', 'J=Vinf/nD', 'CT=T/rho*n2*D4',
                                'CP=P/rho*n3*D5', 'eta=J*CT/CP'],
                       sep=',')

    J_experimental = file['J=Vinf/nD']
    CT_experimental = file['CT=T/rho*n2*D4']
    CP_experimental = file['CP=P/rho*n3*D5']
    eta_experimental = file['eta=J*CT/CP']

    # === Generate numerical data ===
    J_min = np.min(J_experimental)
    J_max = np.max(J_experimental)
    J_numerical = np.linspace(J_min, J_max, 10)
    rot_rate = (wingpropinfo.parameters.vinf /
                (J_numerical*2.*prop_radius)) * 2. * np.pi  # in rad/s

    CT_numerical, CP_numerical = [], []

    prob = om.Problem()

    for index_rotational, irot_rate in enumerate(rot_rate):
        for index_prop in range(wingpropinfo.nr_props):
            wingpropinfo.propeller[index_prop].rot_rate = irot_rate

        prob.model = PropAnalysis(WingPropInfo=wingpropinfo)
        prob.setup()
        prob.run_model()

        n = irot_rate/(2.*np.pi)

        thrust = prob["PropellerModel.HELIX_0.om_helix.rotorcomp_0_thrust"][-1, 0]
        power = prob["PropellerModel.HELIX_0.om_helix.rotorcomp_0_power"][0]
        CT = thrust/(air_density * n**2 * (2*prop_radius)**4)
        CP = power/(air_density * n**3 * (2*prop_radius)**5)

        CT_numerical.append(CT)
        CP_numerical.append(CP)

    # === Plot results ===
    plt.style.use(niceplots.get_style())
    _, ax = plt.subplots(figsize=(10, 7))

    ax.plot(J_numerical, CT_numerical, label='Numerical, CT', color='b')
    ax.plot(J_numerical, CP_numerical, label=f'Numerical, CP', color='orange')

    ax.scatter(J_experimental, CT_experimental,
               label=f'Experimental, CT', color='b')
    ax.scatter(J_experimental, CP_experimental,
               label=f'Experimental, CP', color='orange')

    ax.set_xlabel("Advance Ratio (J)", fontweight='ultralight')
    ax.set_ylabel(r"$C_T$, $C_P$", fontweight='ultralight')
    ax.legend(fontsize='12')

    niceplots.adjust_spines(ax, outward=True)

    plt.savefig(os.path.join(BASE_DIR, 'figures',
                'PROWIM_PROPELLER_VALIDATION.png'))
