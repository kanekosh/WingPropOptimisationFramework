# --- Built-ins ---
from pathlib import Path
import os
import logging
import json

# --- Internal ---
from src.base import ParamInfo, WingPropInfo, WingInfo, PropInfo, AirfoilInfo
from src.utils.tools import print_results
from src.postprocessing.plots import all_plots, stackedplots_wing
from src.integration.coupled_groups_optimisation_new import WingSlipstreamPropOptimisation
# from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

# --- External ---
import openmdao.api as om
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots


logging.getLogger('matplotlib.font_manager').disabled = True
BASE_DIR = Path(__file__).parents[0]

prop_radius = 0.1185
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
wingspan = 0.73*2.*0.952

air_density = 1.2087

spanwise_discretisation_propeller_BEM = len(span)


prop1 = PropInfo(label='Prop1',
                 prop_location=-0.332,
                 nr_blades=4,
                 rot_rate=300.,
                 chord=np.array(chord, order='F')*prop_radius,
                 twist=np.array(twist, order='F'),
                 span=np.array(span, order='F'),
                 airfoils=[AirfoilInfo(label=f'Foil_{index}',
                                       Cl_alpha=Cl_alpha[index],
                                       alpha_L0=alpha_L0[index],
                                       alpha_0=alpha_0[index],
                                       M=M[index])
                           for index in range(spanwise_discretisation_propeller_BEM+1)],
                 ref_point=np.array(ref_point),
                 hub_orientation=np.array([[1.0, 0.0, 0.0],
                                           [0.0, 1.0*np.cos(np.deg2rad(-0.2)),
                                            1.0*np.sin(np.deg2rad(-0.2))],
                                           [0.0, -1.0*np.sin(np.deg2rad(-0.2)), 1.0*np.cos(np.deg2rad(-0.2))]])
                 )

prop2 = PropInfo(label='Prop1',
                 prop_location=0.332,
                 nr_blades=4,
                 rot_rate=300.,
                 chord=np.array(chord, order='F')*prop_radius,
                 twist=np.array(twist, order='F'),
                 span=np.array(span, order='F'),
                 airfoils=[AirfoilInfo(label=f'Foil_{index}',
                                       Cl_alpha=Cl_alpha[index],
                                       alpha_L0=alpha_L0[index],
                                       alpha_0=alpha_0[index],
                                       M=M[index])
                           for index in range(spanwise_discretisation_propeller_BEM+1)],
                 ref_point=ref_point,
                 hub_orientation=np.array([[1.0, 0.0, 0.0],
                                           [0.0, 1.0*np.cos(np.deg2rad(-0.2)),
                                            1.0*np.sin(np.deg2rad(-0.2))],
                                           [0.0, -1.0*np.sin(np.deg2rad(-0.2)), 1.0*np.cos(np.deg2rad(-0.2))]])
                 )

parameters = ParamInfo(vinf=40.,
                       wing_aoa=0.,
                       mach_number=0.2,
                       reynolds_number=640_000,
                       speed_of_sound=333.4,
                       air_density=air_density)


wing = WingInfo(label='PROWIM_wing',
                span=wingspan,
                chord=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*wing_chord,
                twist=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*wing_twist,
                thickness=np.ones(spanwise_discretisation_propeller_BEM,
                              order='F')*0.01,
                empty_weight=0.,
                CL0 = 0.283
                )


PROWIM_wingpropinfo = WingPropInfo(spanwise_discretisation_wing=33,
                            spanwise_discretisation_propeller=21,
                            spanwise_discretisation_propeller_BEM=spanwise_discretisation_propeller_BEM,
                            propeller=[prop1, prop2],
                            wing=wing,
                            parameters=parameters
                            )

if __name__ == '__main__':
    # === Load in (experimental) validation data ===
    validationsetup_file = os.path.join(BASE_DIR, 'data', 'PROWIM_validation_conventional.txt')
    validationsetup_data = pd.read_csv(validationsetup_file, delimiter=',', skiprows=22)
    
    # Validation data for J=inf (prop-off)
    n=0
    index1 = n*19
    index2 = (n+1)*19
    aoa = validationsetup_data['AoA'][index1:index2]
    CL_Jinf = validationsetup_data['CL'][index1:index2]
    CD_Jinf = validationsetup_data['CD'][index1:index2]
    J_inf = validationsetup_data['J'][index1+1]
    
    # Validation data for J=1
    n=2
    index1 = n*19
    index2 = (n+1)*19
    aoa = validationsetup_data['AoA'][index1:index2]
    CL_J1 = validationsetup_data['CL'][index1:index2]
    CD_J1 = validationsetup_data['CD'][index1:index2]
    J_1 = validationsetup_data['J'][index1+1]
    
    # Validation data for J=0.796
    n=3
    index1 = n*19
    index2 = (n+1)*19
    aoa = validationsetup_data['AoA'][index1:index2]
    CL_J0796 = validationsetup_data['CL'][index1:index2]
    CD_J0796 = validationsetup_data['CD'][index1:index2]
    J_0796 = validationsetup_data['J'][index1+1]
    
    # Validation data for J=0.696
    n=4
    index1 = n*19
    index2 = (n+1)*19
    aoa = validationsetup_data['AoA'][index1:index2]
    CL_J0696 = validationsetup_data['CL'][index1:index2]
    CD_J0696 = validationsetup_data['CD'][index1:index2]
    J_0696 = validationsetup_data['J'][index1+1]
    
    rhoinf = validationsetup_data['rhoInf'][0]
    Vinf = validationsetup_data['Vinf'][0]
    qinf = 0.5*rhoinf*Vinf**2
    
    validation_file = os.path.join(BASE_DIR, 'data', 'exp_data_conv_J07.txt')
    validation_CL_CX_data = pd.read_csv(validation_file, delimiter=',')
    CX_validation = validation_CL_CX_data['CX_J=0.7']
    CL_validation = validation_CL_CX_data['CL_J=0.7']

    angles = np.linspace(-5, 11, 6)
    T, D, CL, CX, CD = [], [], [], [], []
    
    # === Setup PROWIM test case ===
    PROWIM_wingpropinfo.wing.empty_weight = 5 # to make T=D
    PROWIM_wingpropinfo.wing.CL0 = 0. # to make T=D
    PROWIM_wingpropinfo.gamma_tangential_dx = 0.3
    PROWIM_wingpropinfo.NO_CORRECTION = False
    PROWIM_wingpropinfo.NO_PROPELLER = False
    
    J = np.array([0.7, 0.796, 0.8960, float('nan')])
    rot_rate = (PROWIM_wingpropinfo.parameters.vinf/(J*2.*prop_radius)) * 2.*np.pi # in rad/s
    
    for iprop, _ in enumerate(PROWIM_wingpropinfo.propeller):
        PROWIM_wingpropinfo.propeller[iprop].rot_rate = rot_rate[0]

    PROWIM_wingpropinfo.__post_init__()
    
    for angle in angles:
        print(f'Angle of attack: {angle: ^10}')
        PROWIM_wingpropinfo.parameters.wing_aoa = angle

        prob = om.Problem()
        prob.model = WingSlipstreamPropOptimisation(WingPropInfo=PROWIM_wingpropinfo,
                                                    objective={},
                                                    constraints={},
                                                    design_vars={})

        # === Analysis ===
        prob.setup()
        prob.run_model()
    
        T = prob['HELIX_COUPLED.thrust_total']
        S_ref = prob['OPENAEROSTRUCT.AS_point_0.wing_perf.S_ref']-0.06995*0.24*2 # correction for nacelle
        D = prob['OPENAEROSTRUCT.AS_point_0.total_perf.D']
        
        CL.append(prob['OPENAEROSTRUCT.AS_point_0.total_perf.CL'])
        CD.append(prob['OPENAEROSTRUCT.AS_point_0.total_perf.CD'])
        cx = (D-T)/(qinf*S_ref)
        CX.append(cx)
    
    plt.style.use(niceplots.get_style())
    _, ax = plt.subplots(figsize=(10, 7))
    
    # ax.plot(angles, T, label=r'$T$', color='b')
    # ax.plot(angles, D, label=r'$D$', color='orange')
    ax.scatter(CX_validation, CL_validation, label='Experimental data', color='orange')
    ax.plot(CX, CL, label=r'$C_L$', color='b')
    ax.set_ylabel(r'$C_L (-)$')
    ax.set_xlabel(r'$C_X (-)$')
    # ax.set_ylim((-0.4, 1.6))
    # ax.set_xlim((-0.2, 0.))
    ax.legend()
    
    niceplots.adjust_spines(ax, outward=True)
    
    plt.savefig(os.path.join(BASE_DIR, 'figures', 'TUBE_PROWIM_VALIDATION.png'))
