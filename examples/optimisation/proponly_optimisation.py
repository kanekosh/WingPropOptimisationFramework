# --- Built-ins ---
from pathlib import Path
import os
import logging
import copy

# --- Internal ---
from src.utils.tools import print_results
from src.postprocessing.plots import all_plots
from src.integration.coupled_groups import PropOptimisation
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo, PROWIM_prop_1, PROWIM_parameters

# --- External ---
import openmdao.api as om
import numpy as np


logging.getLogger('matplotlib.font_manager').disabled = True
BASE_DIR = Path(__file__).parents[0]

if __name__=='__main__':
    PROWIM_wingpropinfo.propeller = [PROWIM_prop_1]
    
    objective = {
                'HELIX_COUPLED.power_total':
                    {'scaler': 1/689.2153704}
                }

    design_vars = {
                    'PropellerModel.om_helix.geodef_parametric_0_twist':
                        {'lb': 0,
                        'ub': 90,
                        'scaler': 1./10},
                    'PropellerModel.om_helix.geodef_parametric_0_rot_rate':
                        {'lb': 0,
                        'ub': 3000,
                        'scaler': 1./2300},
                    # 'PropellerModel.om_helix.rotor_0_twist':
                    #     {'lb': 0,
                    #     'ub': 90,
                    #     'scaler': 1./10},
                    # 'PropellerModel.om_helix.rotor_1_twist':
                    #     {'lb': 0,
                    #     'ub': 90,
                    #     'scaler': 1./10},
                    }

    constraints = {
                    'HELIX_COUPLED.thrust_total':
                        {'equals': 15.22511642}
                    }
    
    prob = om.Problem()
    prob.model = PropOptimisation(ParamInfo=PROWIM_parameters,
                                    PropInfo=PROWIM_prop_1,
                                    WingPropInfo=PROWIM_wingpropinfo,
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
    "Major feasibility tolerance": 1.0e-5,
    "Major optimality tolerance": 1.0e-5,
    "Minor feasibility tolerance": 1.0e-5,
    "Verify level": -1,
    "Function precision": 1.0e-6,
    # "Major iterations limit": 50,
    "Nonderivative linesearch": None,
    "Print file": os.path.join(BASE_DIR, 'results', 'optimisation_print.out'),
    "Summary file": os.path.join(BASE_DIR, 'results', 'optimisation_summary.out')
    }
    
    print(prob['HELIX_COUPLED.thrust_total'], prob['HELIX_COUPLED.power_total'])
    
        # Initialise recorder
    db_name = os.path.join(BASE_DIR, 'results', 'data_propeller.db')
    
    recorder = om.SqliteRecorder(db_name)
    prob.driver.add_recorder(recorder)
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['includes'] = [
                                                    "PropellerModel.om_helix.rotorcomp_0_radii",
                                                    "PropellerModel.om_helix.rotorcomp_0_velocity_distribution"
                                                 ]
    
    print('==========================================================')
    print('====================== Optimisation ======================')
    print('==========================================================')
    prob.setup()
    prob.run_driver()
    
    print_results(design_vars=design_vars, constraints=constraints, objective=objective,
                  prob=prob, kind="Initial Analysis")
    
    savepath = os.path.join(BASE_DIR, 'results', 'prop_results')
    all_plots(db_name=db_name,
              wingpropinfo=PROWIM_wingpropinfo,
              savedir=savepath)