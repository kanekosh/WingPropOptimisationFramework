# --- Built-ins ---
from pathlib import Path
import os
import logging
import copy

# --- Internal ---
from src.utils.tools import print_results
from src.postprocessing.plots import all_plots, stackedplots_wing
from src.integration.coupled_groups_optimisation import WingOptimisation
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

# --- External ---
import openmdao.api as om
import numpy as np


logging.getLogger('matplotlib.font_manager').disabled = True
BASE_DIR = Path(__file__).parents[0]

if __name__ == '__main__':
    
    # db_name = os.path.join(BASE_DIR, 'results', 'data_wing.db')
    # savepath = os.path.join(BASE_DIR, 'results', 'wing_results')
    # stackedplots_wing(db_name=db_name,
    #             wingpropinfo=PROWIM_wingpropinfo,
    #             savedir=savepath,
    #             noprop=True)
    # quit()

    PROWIM_wingpropinfo.wing.empty_weight = 4 # to make T=D
    PROWIM_wingpropinfo.wing.CL0 = 0. # to make T=D
    # PROWIM_wingpropinfo.wing.fuel_mass = 0 # to make T=D
    # PROWIM_wingpropinfo.wing.span *= 1
    # PROWIM_wingpropinfo.wing.chord *= 2
    PROWIM_wingpropinfo.linear_mesh = True # smoothness of function is determined by this
    
    # PROWIM_wingpropinfo.wing.twist = np.array([-0.31981993,  1.50942036,  3.5180305,   5.75997867,  6.99893059,  6.99893087, 5.75997605,  3.51803336,  1.50942123, -0.3198214])
    # PROWIM_wingpropinfo.wing.chord =  np.array([0.08378153, 0.08269444, 0.08499097, 0.08143507, 0.0927921,  0.09279211, 0.08143507, 0.08499097, 0.08269444, 0.08378153])
    
    PROWIM_wingpropinfo.__post_init__()
    
    objective = {
                'OPENAEROSTRUCT.AS_point_0.total_perf.D':
                    {'scaler': 1/9.81879759}
                }

    design_vars = {
                    'OPENAEROSTRUCT.wing.twist_cp':
                        {'lb': -10,
                        'ub': 8,
                        'scaler': 1},
                    'OPENAEROSTRUCT.wing.geometry.chord_cp':
                        {'lb': 0.01,
                        'ub': 30,
                        'scaler': 1},
                    'OPENAEROSTRUCT.wing.thickness_cp':
                        {'lb': 5e-3,
                        'ub': 2e-1,
                        'scaler': 1e2},
                    # 'PARAMETERS.alpha':
                    #     {'lb': -10,
                    #     'ub': 8,
                    #     'scaler': 1},
                    }

    constraints = {
                    'OPENAEROSTRUCT.AS_point_0.total_perf.CL':
                        {'upper': 0.8},
                    'OPENAEROSTRUCT.AS_point_0.L_equals_W':
                        {'equals': 0.},
                    'OPENAEROSTRUCT.AS_point_0.wing_perf.failure':
                        {'upper': 0.},
                    'OPENAEROSTRUCT.AS_point_0.wing_perf.thickness_intersects':
                        {'upper': 0.},
                    'OPENAEROSTRUCT.wing.structural_mass':
                        {'lower': 0.},
                    }
    
    prob = om.Problem()
    prob.model = WingOptimisation(  WingPropInfo=PROWIM_wingpropinfo,
                                    objective=objective,
                                    constraints=constraints,
                                    design_vars=design_vars)
    
    # === Analysis ===
    prob.setup()
    prob.run_model()
    om.n2(prob, 'wingonly_opt.html')
    
    print_results(design_vars=design_vars, constraints=constraints, objective=objective,
                  prob=prob, kind="Initial Analysis")

    # === Optimisation ===
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs']
    prob.driver.opt_settings = {
    "Major feasibility tolerance": 1.0e-6,
    "Major optimality tolerance": 1.0e-6,
    "Minor feasibility tolerance": 1.0e-6,
    "Verify level": -1,
    "Function precision": 1.0e-6,
    # "Major iterations limit": 2,
    "Nonderivative linesearch": None,
    "Print file": os.path.join(BASE_DIR, 'results', 'optimisation_print_wing.out'),
    "Summary file": os.path.join(BASE_DIR, 'results', 'optimisation_summary_wing.out')
    }
    
        # Initialise recorder
    db_name = os.path.join(BASE_DIR, 'results', 'data_wing.db')
    
    recorder = om.SqliteRecorder(db_name)
    prob.driver.add_recorder(recorder)
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['includes'] = [
                                                    # "OPENAEROSTRUCT.wing.mesh",
                                                    "OPENAEROSTRUCT.wing.geometry.twist",
                                                    "OPENAEROSTRUCT.wing.geometry.chord",
                                                    "OPENAEROSTRUCT.AS_point_0.wing_perf.Cl",
                                                    "OPENAEROSTRUCT.AS_point_0.total_perf.L",
                                                    "OPENAEROSTRUCT.AS_point_0.total_perf.D",
                                                ]
    
    print('==========================================================')
    print('====================== Optimisation ======================')
    print('==========================================================')
    prob.setup()
    prob.run_driver()
    
    print_results(design_vars=design_vars, constraints=constraints, objective=objective,
                  prob=prob, kind="Optimisation")
    
    savepath = os.path.join(BASE_DIR, 'results', 'wing_results')
    all_plots(db_name=db_name,
              wingpropinfo=PROWIM_wingpropinfo,
              savedir=savepath)
    
    stackedplots_wing(db_name=db_name,
                wingpropinfo=PROWIM_wingpropinfo,
                savedir=savepath,
                noprop=True)