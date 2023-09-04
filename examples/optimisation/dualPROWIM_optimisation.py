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
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

# --- External ---
import openmdao.api as om
import numpy as np


logging.getLogger('matplotlib.font_manager').disabled = True
BASE_DIR = Path(__file__).parents[0]

if __name__ == '__main__':
    objective = {'PropellerSlipstreamWingModel.HELIX_COUPLED.power_total':
                {'scaler': 1/1_000}}

    design_vars = {'PropellerSlipstreamWingModel.DESIGNVARIABLES.rotor_0_rot_rate':
                        {'lb': 0,
                        'ub': 3000,
                        'scaler': 1./1_000},
                    'PropellerSlipstreamWingModel.DESIGNVARIABLES.rotor_1_rot_rate':
                        {'lb': 0,
                        'ub': 3000,
                        'scaler': 1./1_000},
                    'PropellerSlipstreamWingModel.DESIGNVARIABLES.twist':
                        {'lb': -5,
                        'ub': 5,
                        'scaler': 1},
                    }

    constraints = {'PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.CL':
                        {'equals': 0.35},
                    'PropellerSlipstreamWingModel.CONSTRAINTS.thrust_equals_drag':
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
                                    includes=['*PropellerSlipstreamWingModel.RETHORST*'], 
                                    form='central', step=1e-8)

    print_results(design_vars=design_vars, constraints=constraints, objective=objective,
                  prob=prob, kind="Initial Analysis")

    # === Optimisation ===
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.opt_settings = {
        "MAXIT": 150,
        'IFILE': os.path.join(BASE_DIR, 'results', 'optimisation_log.out')
    }
    
        # Initialise recorder
    db_name = os.path.join(BASE_DIR, 'results', 'data.db')
    savepath = os.path.join(BASE_DIR, 'results')
    
    recorder = om.SqliteRecorder(db_name)
    prob.driver.add_recorder(recorder)
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['includes'] = ["PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.Cl",
                                                 'PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.total_perf.D']
    
    print('==========================================================')
    print('====================== Optimisation ======================')
    print('==========================================================')
    prob.setup()
    prob.run_driver()
    
    prob.cleanup() # close all recorders

    print_results(design_vars=design_vars, constraints=constraints, objective=objective,
                  prob=prob, kind="Results")
    
    # === Plotting ===
    savepath = os.path.join(BASE_DIR, 'results')
    all_plots(db_name=db_name,
              wingpropinfo=PROWIM_wingpropinfo,
              savedir=savepath)
