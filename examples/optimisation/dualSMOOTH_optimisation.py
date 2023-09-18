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
    # === Plotting ===
    # db_name = os.path.join(BASE_DIR, 'results', 'data_wingprop.db')
    # savepath = os.path.join(BASE_DIR, 'results', 'propwing_results')
    # stackedplots_prop(db_name=db_name,
    #                 wingpropinfo=PROWIM_wingpropinfo,
    #                 savedir=savepath)
    # # stackedplots_wing(db_name=db_name,
    # #             wingpropinfo=PROWIM_wingpropinfo,
    # #             savedir=savepath)
    # quit()

    PROWIM_wingpropinfo.wing.empty_weight = 4 # to make T=D
    PROWIM_wingpropinfo.wing.CL0 = 0. # to make T=D
    
    # PROWIM_wingpropinfo.wing.twist = np.array([0.16758064, 0.16531812, 0.17043133, 0.16231965, 0.18910957,
    #    0.18910647, 0.16232025, 0.17043081, 0.16531831, 0.16758051])
    # PROWIM_wingpropinfo.wing.chord = np.array([0.16758064, 0.16531812, 0.17043133, 0.16231965, 0.18910957,
    #    0.18910647, 0.16232025, 0.17043081, 0.16531831, 0.16758051])
    
    # for iprop, _ in enumerate(PROWIM_wingpropinfo.propeller):
    #     PROWIM_wingpropinfo.propeller[iprop].twist = np.array([86.03333075, 84.34827976, 82.77931173, 81.35095221, 79.25326958,
    #                                                             79.14545502, 78.52304526, 76.11417496, 75.71005392, 74.84041569,
    #                                                             71.90351263, 71.62839411, 70.53516045, 67.64705164, 67.30649674,
    #                                                             66.0399728 , 63.37655562, 62.88408362, 61.5509477 , 57.92588983])
    #     PROWIM_wingpropinfo.propeller[iprop].rot_rate = 222.56

    PROWIM_wingpropinfo.__post_init__()
    
    objective = {
                'HELIX_COUPLED.power_total':
                    {'scaler': 1/433.04277037}
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
                        {'upper': 1.},
                    'OPENAEROSTRUCT.AS_point_0.wing_perf.thickness_intersects':
                        {'upper': 0.},
                    'OPENAEROSTRUCT.AS_point_0.L_equals_W':
                        {'equals': 0.},
                    'CONSTRAINTS.thrust_equals_drag':
                        {'equals': 0.}
                    }
    
    # optimisation_arch = MainWingPropOptimisation(wingpropinfo=PROWIM_wingpropinfo,
    #                                              objective=objective,
    #                                              constraints=constraints,
    #                                              design_variables=design_vars,
    #                                              database_savefile='.',
    #                                              result_dir='.')
    
    
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
    prob.driver.options['debug_print'] = ['desvars', 'nl_cons']
    prob.driver.opt_settings = {
        "Major feasibility tolerance": 1.0e-6,
        "Major optimality tolerance": 1.0e-6,
        "Minor feasibility tolerance": 1.0e-6,
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
                "HELIX_0.om_helix.geodef_parametric_0_rot_rate"]
    
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