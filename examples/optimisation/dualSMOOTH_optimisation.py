# --- Built-ins ---
from pathlib import Path
import os
import logging
import copy

# --- Internal ---
from src.utils.tools import print_results
from src.postprocessing.plots import all_plots, stackedplots_wing, stackedplots_prop
from src.integration.coupled_groups_optimisation import WingSlipstreamPropOptimisation
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

# --- External ---
import openmdao.api as om
import numpy as np


logging.getLogger('matplotlib.font_manager').disabled = True
BASE_DIR = Path(__file__).parents[0]

if __name__ == '__main__':
    # Start from 
    PROWIM_wingpropinfo.wing.empty_weight = 2 # to make T=D
    PROWIM_wingpropinfo.wing.CL0 = 0. # to make T=D
    # PROWIM_wingpropinfo.spanwise_discretisation_propeller = 21
    
    PROWIM_wingpropinfo.wing.twist = np.array([-1.25865721, -0.45725257,  0.65422908,  1.70931064,  2.57777181,
                                                3.05298624,  3.39002958,  3.0529862 ,  2.57777187,  1.70931056,
                                                0.65422926, -0.45725286, -1.25865693])
    PROWIM_wingpropinfo.wing.chord =  np.array([0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.08333333,
                                                0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.08333333,
                                                0.08333333, 0.08333333, 0.0833333])
    
    for iprop, _ in enumerate(PROWIM_wingpropinfo.propeller):
        PROWIM_wingpropinfo.propeller[iprop].rot_rate = 230.0476
        PROWIM_wingpropinfo.propeller[iprop].twist = np.array([ 85.83994569, 84.13571425, 82.42911816, 80.77826907, 79.01097771, 78.61916379,
                                                                77.61554485, 76.44706709, 75.07820051, 73.72439922, 72.31022309, 70.84585741,
                                                                69.38616929, 67.9114225,  66.43057136, 64.94354747, 63.42398014, 61.82089418,
                                                                60.09301686, 57.8621155])
                                                                            
    # === Plotting ===
    # db_name = os.path.join(BASE_DIR, 'results', 'data_wingprop.db')
    # savepath = os.path.join(BASE_DIR, 'results', 'propwing_results')
    # stackedplots_prop(db_name=db_name,
    #                 wingpropinfo=PROWIM_wingpropinfo,
    #                 savedir=savepath)
    # stackedplots_wing(db_name=db_name,
    #             wingpropinfo=PROWIM_wingpropinfo,
    #             savedir=savepath)
    # quit()
    
    objective = {
                'HELIX_COUPLED.power_total':
                    {'scaler': 1/(2*32.17610257)}
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
                    # 'PARAMETERS.alpha':
                    #     {'lb': -10,
                    #     'ub': 8,
                    #     'scaler': 1},
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
                        {'upper': 0.8},
                    'OPENAEROSTRUCT.AS_point_0.wing_perf.thickness_intersects':
                        {'upper': 0.},
                    'OPENAEROSTRUCT.AS_point_0.L_equals_W':
                        {'equals': 0.},
                    'HELIX_COUPLED.thrust_total':
                        {'lower': 0.},
                    'CONSTRAINTS.thrust_equals_drag':
                        {'equals': 0.},
                    'OPENAEROSTRUCT.wing.structural_mass':
                        {'lower': 0.},
                    'OPENAEROSTRUCT.AS_point_0.total_perf.D':
                        {'lower': 0.}
                    }

    prob = om.Problem()
    prob.model = WingSlipstreamPropOptimisation(WingPropInfo=PROWIM_wingpropinfo,
                                                objective=objective,
                                                constraints=constraints,
                                                design_vars=design_vars)

    # === Analysis ===
    prob.setup()
    prob.run_model()

        # Check derivatives  
    if False:
        prob.check_totals(  compact_print=True, show_only_incorrect=True,
                        form='central', step=1e-8, 
                        rel_err_tol=1e-3)
        # prob.check_partials(compact_print=True, show_only_incorrect=True, 
        #                             excludes=['*HELIX_0*', '*HELIX_1*'], 
        #                             form='central', step=1e-8)
        
        quit()

    print_results(design_vars=design_vars, constraints=constraints, objective=objective,
                  prob=prob, kind="Initial Analysis")

    # === Optimisation ===
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs']
    prob.driver.opt_settings = {
        "Major feasibility tolerance": 1.0e-5,
        "Major optimality tolerance": 1.0e-5,
        "Minor feasibility tolerance": 1.0e-5,
        "Verify level": -1,
        "Function precision": 1.0e-6,
        # "Major iterations limit": 1,
        "Nonderivative linesearch": None,
        "Print file": os.path.join(BASE_DIR, 'results', 'optimisation_print_wingprop.out'),
        "Summary file": os.path.join(BASE_DIR, 'results', 'optimisation_summary_wingprop.out')
    }
    
    # prob.check_totals(  compact_print=True, show_only_incorrect=True,
    #                     form='central', step=1e-8, 
    #                     rel_err_tol=1e-3)
    
        # Initialise recorder
    db_name = os.path.join(BASE_DIR, 'results', 'data_wingprop.db')
    
    includes = ["OPENAEROSTRUCT.wing.geometry.twist",
                "OPENAEROSTRUCT.wing.geometry.chord",
                "OPENAEROSTRUCT.AS_point_0.wing_perf.Cl",
                "OPENAEROSTRUCT.AS_point_0.wing_perf.CDi",
                'OPENAEROSTRUCT.AS_point_0.total_perf.L',
                'OPENAEROSTRUCT.AS_point_0.total_perf.D',
                'RETHORST.velocity_distribution',
                'RETHORST.propeller_velocity',
                "HELIX_0.om_helix.rotorcomp_0_radii",
                "HELIX_0.om_helix.rotorcomp_0_velocity_distribution",
                "HELIX_0.om_helix.geodef_parametric_0_twist",
                "HELIX_0.om_helix.geodef_parametric_0_rot_rate",
                "PARAMETERS.wing_mesh",
                "PARAMETERS.wing_mesh_control_points",
                'OPENAEROSTRUCT.wing.structural_mass']
    
    for key in design_vars.keys():
        includes.extend(key)
    for key in design_vars.keys():
        includes.extend(key)
    for key in design_vars.keys():
        includes.extend(key)
    
    recorder = om.SqliteRecorder(db_name)
    prob.driver.add_recorder(recorder)
    prob.driver.add_recorder(recorder)
    # TODO: write code that checks whether the problem variable exists
    prob.driver.recording_options['includes'] = includes
    
    print('==========================================================')
    print('====================== Optimisation ======================')
    print('==========================================================')
    prob.setup()
    prob.run_driver()
    
    prob.cleanup() # close all recorders

    print_results(design_vars=design_vars, constraints=constraints, objective=objective,
                  prob=prob, kind="Results")
    
    # === Plotting ===
    savepath = os.path.join(BASE_DIR, 'results', 'propwing_results')
    all_plots(db_name=db_name,
              wingpropinfo=PROWIM_wingpropinfo,
              savedir=savepath)
    stackedplots_wing(db_name=db_name,
                wingpropinfo=PROWIM_wingpropinfo,
                savedir=savepath)
    stackedplots_prop(db_name=db_name,
                    wingpropinfo=PROWIM_wingpropinfo,
                    savedir=savepath)