# --- Built-ins ---
from pathlib import Path
import os
import logging
import copy

# --- Internal ---
from src.base import WingPropInfo
from src.utils.tools import print_results
from src.postprocessing.plots import all_plots
from src.integration.coupled_groups_optimisation import WingSlipstreamPropOptimisation
from src.integration.wingprop_optimisation import MainWingPropOptimisation
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

# --- External ---
import openmdao.api as om
import numpy as np


logging.getLogger('matplotlib.font_manager').disabled = True
BASE_DIR = Path(__file__).parents[0]

if __name__ == '__main__':
    # Adjustment to the PROWIM setup
    PROWIM_wingpropinfo.spanwise_discretisation_propeller = 25 # to make T=D
    PROWIM_wingpropinfo.__post_init__()

    PROWIM_wingpropinfo.wing.chord *= 18 # to make T=D
    PROWIM_wingpropinfo.wing.twist += 2 # to make T=D
    PROWIM_wingpropinfo.wing.vinf = 40 # to make T=D
    for index in range(len(PROWIM_wingpropinfo.propeller)):
        PROWIM_wingpropinfo.propeller[index].rot_rate = 644.82864419
        PROWIM_wingpropinfo.propeller[index].twist = np.array([67.79378385, 71.83797648, 61.32955902, 62.9787903,  56.87134101, 58.16629045,
                                                                56.66413092, 54.56196904, 52.76508122, 50.14207845, 48.77576388, 45.81754819,
                                                                44.61299923, 42.01886426, 40.93763764, 38.52984867, 37.65342321, 35.1964771,
                                                                33.97829724, 30.47284116],
                                                                    order='F'
                                                            )
    
    
    objective = {
                'HELIX_COUPLED.power_total':
                    {'scaler': 1/654}
                }

    design_vars = {
                    'DESIGNVARIABLES.rotor_0_rot_rate':
                        {'lb': 0,
                        'ub': 3000,
                        'scaler': 1./644.82864419},
                    'DESIGNVARIABLES.rotor_1_rot_rate':
                        {'lb': 0,
                        'ub': 3000,
                        'scaler': 1./644.82864419},
                    'DESIGNVARIABLES.rotor_0_twist':
                        {'lb': 0,
                        'ub': 90,
                        'scaler': 1./10},
                    'DESIGNVARIABLES.rotor_1_twist':
                        {'lb': 0,
                        'ub': 90,
                        'scaler': 1./10},
                    # 'DESIGNVARIABLES.rotor_0_chord':
                    #     {'lb': 0,
                    #     'ub': 90,
                    #     'scaler': 1./10},
                    # 'DESIGNVARIABLES.rotor_1_chord':
                    #     {'lb': 0,
                    #     'ub': 90,
                    #     'scaler': 1./10},
                    'DESIGNVARIABLES.twist':
                        {'lb': -5,
                        'ub': 5,
                        'scaler': 1},
                    'OPENAEROSTRUCT.wing.thickness_cp':
                        {'lb': 1e-3,
                        'ub': 5e-1,
                        'scaler': 1e3},
                    }

    constraints = {
                    'OPENAEROSTRUCT.AS_point_0.wing_perf.failure':
                        {'upper': 0.},
                    'OPENAEROSTRUCT.AS_point_0.wing_perf.thickness_intersects':
                        {'upper': 0.},
                    'OPENAEROSTRUCT.AS_point_0.wing_perf.CL':
                        {'equals': 0.41659956},
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
    model = prob.model
    # === Analysis ===
    prob.setup()
    prob.run_model()
    
        # Check derivatives  
    if False:
        prob.check_totals(  compact_print=True, show_only_incorrect=True,
                        form='central', step=1e-8, 
                        rel_err_tol=1e-3, abs_err_tol=1e-4)
        partials = prob.check_partials(compact_print=True, show_only_incorrect=True, 
                                    includes=['*RETHORST*'], 
                                    form='central', step=1e-8)

    print_results(design_vars=design_vars, constraints=constraints, objective=objective,
                  prob=prob, kind="Initial Analysis")

    # === Optimisation ===
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings = {
        "Major feasibility tolerance": 1.0e-8,
        "Major optimality tolerance": 1.0e-8,
        "Minor feasibility tolerance": 1.0e-8,
        "Verify level": -1,
        "Function precision": 1.0e-6,
        # "Major iterations limit": 1,
        "Nonderivative linesearch": None,
        "Print file": os.path.join(BASE_DIR, 'results', 'optimisation_print_wingprop.out'),
        "Summary file": os.path.join(BASE_DIR, 'results', 'optimisation_summary_wingprop.out')
    }
    
        # Initialise recorder
    db_name = os.path.join(BASE_DIR, 'results', 'data.db')
    
    recorder = om.SqliteRecorder(db_name)
    prob.driver.add_recorder(recorder)
    prob.driver.add_recorder(recorder)
    # TODO: write code that checks whether the problem variable exists
    prob.driver.recording_options['includes'] = ["OPENAEROSTRUCT.wing.geometry.twist",
                                                 "OPENAEROSTRUCT.AS_point_0.wing_perf.Cl",
                                                 "OPENAEROSTRUCT.AS_point_0.wing_perf.CDi",
                                                 'OPENAEROSTRUCT.AS_point_0.total_perf.L',
                                                 'OPENAEROSTRUCT.AS_point_0.total_perf.D',
                                                 'OPENAEROSTRUCT.AS_point_0.total_perf.D',
                                                 'RETHORST.velocity_distribution']
    
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
