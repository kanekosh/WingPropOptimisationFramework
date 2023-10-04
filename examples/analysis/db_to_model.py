# --- Built-ins ---
import os
from pathlib import Path
import json
import logging

# --- Internal ---
from src.base import ParamInfo, WingPropInfo, WingInfo, PropInfo, AirfoilInfo
from src.utils.tools import print_results
from src.integration.coupled_groups_analysis import PropAnalysis
from src.integration.coupled_groups_optimisation_new import WingRethorstPropOptimisation, WingOptimisation, PropOptimisation

# --- External ---
from openmdao.recorders.sqlite_reader import SqliteCaseReader
import numpy as np
import pandas as pd
import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots


logging.getLogger('matplotlib.font_manager').disabled = True
BASE_DIR = Path(__file__).parents[0]

def db_to_wingprop(wingpropinfo: WingPropInfo,
                   objective: dict,
                   constraints: dict,
                   design_vars: dict):
    prob = om.Problem()
    prob.model = WingRethorstPropOptimisation(      WingPropInfo=wingpropinfo,
                                                    objective=objective,
                                                    constraints=constraints,
                                                    design_vars=design_vars)

    # === Analysis ===
    prob.setup()
    prob.run_model()

    print_results(  design_vars=design_vars, 
                    constraints=constraints, 
                    objective=objective,
                    prob=prob, kind="Results"
                )
    

def db_to_prop(wingpropinfo: WingPropInfo,
                objective: dict,
                constraints: dict,
                design_vars: dict):
    prob = om.Problem()
    prob.model = PropOptimisation(      WingPropInfo=wingpropinfo,
                                                    objective=objective,
                                                    constraints=constraints,
                                                    design_vars=design_vars)

    # === Analysis ===
    prob.setup()
    prob.run_model()

    print_results(  design_vars=design_vars, 
                    constraints=constraints, 
                    objective=objective,
                    prob=prob, kind="Results"
                )
    

def db_to_wing(wingpropinfo: WingPropInfo,
                objective: dict,
                constraints: dict,
                design_vars: dict):
    prob = om.Problem()
    prob.model = WingOptimisation(      WingPropInfo=wingpropinfo,
                                                    objective=objective,
                                                    constraints=constraints,
                                                    design_vars=design_vars)

    # === Analysis ===
    prob.setup()
    prob.run_model()
    
    print_results(  design_vars=design_vars, 
                    constraints=constraints, 
                    objective=objective,
                    prob=prob, kind="Results"
                )
    

if __name__=='__main__':
    # === Get optimized geometry ===
    db_name = os.path.join(BASE_DIR, '..', 'optimisation', 'results', 'data_propeller.db')
    database = SqliteCaseReader(db_name, pre_load=True)
    optimised = database.get_cases()[-1]

    try:
        wing_twist = optimised.outputs['DESIGNVARIABLES.twist']
        wing_chord = optimised.outputs['DESIGNVARIABLES.chord']*0.24
    except:
        wing_twist = 0
        wing_chord = 0
    
    try:
        prop_twist = optimised.outputs['DESIGNVARIABLES.rotor_0_twist']
        rot_rate = 0.5*(optimised.outputs['DESIGNVARIABLES.rotor_0_rot_rate']+optimised.outputs['DESIGNVARIABLES.rotor_1_rot_rate'])
    except:
        prop_twist = np.zeros(20)
        rot_rate = 0
        
    
    # === Read in PROWIM data ===
    with open(os.path.join(BASE_DIR, 'data', 'PROWIM.json'), 'r') as file:
        data = json.load(file)

    prop_radius = 0.1185
    ref_point = data['ref_point']
    span = data['span']
    chord = data['chord']

    alpha_0 = data['alpha_0']
    alpha_L0 = data['alpha_L0']
    Cl_alpha = data['Cl_alpha']
    M = data['M']
    
    # === Parameters ===
    prop_refinement = 4
    spanwise_discretisation_propeller_BEM = len(span)
    wingspan = 0.73*2.*0.952

    PROWIM_parameters = ParamInfo(  vinf=40.,
                                    wing_aoa=2., # TODO: this is a wing property
                                    mach_number=0.2,
                                    reynolds_number=3_500_000,
                                    speed_of_sound=333.4,
                                    air_density=1.2087)

    PROWIM_prop_1 = PropInfo(   label='Prop1',
                                prop_location=-0.332,
                                nr_blades=4,
                                rot_rate=rot_rate,
                                chord=np.array(chord, order='F')*prop_radius,
                                twist=np.array(prop_twist, order='F'),
                                span=np.array(span, order='F'),
                                airfoils=[AirfoilInfo(  label=f'Foil_{index}',
                                                        Cl_alpha=Cl_alpha[index],
                                                        alpha_L0=alpha_L0[index],
                                                        alpha_0=alpha_0[index],
                                                        M=M[index])
                                                        for index in range(spanwise_discretisation_propeller_BEM+1)],
                                ref_point=ref_point,
                                local_refinement=prop_refinement,
                                rotation_direction=-1,
                                prop_angle=45
                                )

    PROWIM_prop_2 = PropInfo(label='Prop1',
                    prop_location=0.332,
                    nr_blades=4,
                    rot_rate=rot_rate,
                    chord=np.array(chord, order='F')*prop_radius,
                    twist=np.array(prop_twist, order='F'),
                    span=np.array(span, order='F'),
                    airfoils=[AirfoilInfo(  label=f'Foil_{index}',
                                            Cl_alpha=Cl_alpha[index],
                                            alpha_L0=alpha_L0[index],
                                            alpha_0=alpha_0[index],
                                            M=M[index])
                                            for index in range(spanwise_discretisation_propeller_BEM+1)],
                    ref_point=ref_point,
                    local_refinement=prop_refinement,
                    rotation_direction=1,
                    prop_angle=45
                    )


    PROWIM_wing = WingInfo(label='PROWIM_wing',
                    span=wingspan,
                    thickness = np.ones(10)*0.003,
                    chord=np.array(wing_chord),
                    twist=np.array(wing_twist),
                    empty_weight=5.,
                    CL0 = 0.0, # if you want to do optimization set this to zero bcs otherwise OAS will return erroneous results
                    fuel_mass=0
                    )
    
    wingpropinfo = WingPropInfo(    spanwise_discretisation_propeller=21,
                                    spanwise_discretisation_propeller_BEM=spanwise_discretisation_propeller_BEM,
                                    propeller=[PROWIM_prop_1, PROWIM_prop_2],
                                    wing=PROWIM_wing,
                                    parameters=PROWIM_parameters
                                )
    
    objective = {
                # 'HELIX_COUPLED.power_total':
                #     {'scaler': 1/433.04277037}
                }

    design_vars = {
                    'DESIGNVARIABLES.rotor_0_rot_rate':
                        {'lb': 0,
                        'ub': 3000,
                        'scaler': 1./1060},
                    'DESIGNVARIABLES.rotor_1_rot_rate':
                        {'lb': 0,
                        'ub': 3000,
                        'scaler': 1./1060},
                    'DESIGNVARIABLES.rotor_0_twist':
                        {'lb': 0,
                        'ub': 90,
                        'scaler': 1./10},
                    'DESIGNVARIABLES.rotor_1_twist':
                        {'lb': 0,
                        'ub': 90,
                        'scaler': 1./10},
                    'DESIGNVARIABLES.twist':
                        {'lb': -10,
                        'ub': 8,
                        'scaler': 1},
                    'DESIGNVARIABLES.chord':
                        {'lb': 0,
                        'ub': 3,
                        'scaler': 1},
                    'OPENAEROSTRUCT.wing.thickness_cp':
                        {'lb': 3e-3,
                        'ub': 5e-1,
                        'scaler': 1e2},
                    }

    constraints = {
                    'OPENAEROSTRUCT.AS_point_0.wing_perf.failure':
                        {'upper': 0.},
                    'OPENAEROSTRUCT.AS_point_0.total_perf.CL':
                        {'upper': 0.9},
                    'OPENAEROSTRUCT.AS_point_0.wing_perf.thickness_intersects':
                        {'upper': 0.},
                    'OPENAEROSTRUCT.AS_point_0.L_equals_W':
                        {'equals': 0.},
                    # 'CONSTRAINTS.thrust_equals_drag':
                    #     {'equals': 0.}
                    }
    
    db_to_wingprop( wingpropinfo=wingpropinfo,
                    objective=objective,
                    design_vars=design_vars,
                    constraints=constraints)