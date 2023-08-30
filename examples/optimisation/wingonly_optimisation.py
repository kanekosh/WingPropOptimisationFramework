# --- Built-ins ---
from pathlib import Path
import os
import logging
import copy

# --- Internal ---
from src.utils.tools import print_results
from src.postprocessing.plots import all_plots
from src.integration.coupled_groups_optimisation import WingOptimisation
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

# --- External ---
import openmdao.api as om
import numpy as np


logging.getLogger('matplotlib.font_manager').disabled = True
BASE_DIR = Path(__file__).parents[0]

if __name__ == '__main__':
    PROWIM_wingpropinfo.wing.twist = np.ones(13)*2
    
    objective = {
                'OPENAEROSTRUCT.AS_point_0.wing_perf.CDi':
                    {'scaler': 1/0.00217305}
                }

    design_vars = {
                    'DESIGNVARIABLES.twist':
                        {'lb': -5,
                        'ub': 5,
                        'scaler': 1},
                    }

    constraints = {
                    'OPENAEROSTRUCT.AS_point_0.wing_perf.CL':
                        {'equals': 0.66805657},
                    }
    
    prob = om.Problem()
    prob.model = WingOptimisation(  WingPropInfo=PROWIM_wingpropinfo,
                                    objective=objective,
                                    constraints=constraints,
                                    design_vars=design_vars)
    
    prob.setup()
    prob.run_model()
    
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
    # "Major iterations limit": 50,
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
                                                    "OPENAEROSTRUCT.wing.geometry.twist",
                                                    "OPENAEROSTRUCT.AS_point_0.wing_perf.Cl"
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