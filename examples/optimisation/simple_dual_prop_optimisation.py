# --- Built-ins ---
from pathlib import Path
import os
import logging
import copy

# --- Internal ---
from src.base import ParamInfo, WingPropInfo, WingInfo, PropInfo, AirfoilInfo
from src.tools.tools import print_results, plot_CL
from src.integration.model_coupling import WingSlipstreamProp
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

# --- External ---
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots


logging.getLogger('matplotlib.font_manager').disabled = True
BASE_DIR = Path(__file__).parents[0]

objective = {'PropellerSlipstreamWingModel.HELIX_COUPLED.power_total':
                {'scaler': 1000}}
design_vars = {'PropellerSlipstreamWingModel.DESIGNVARIABLES.twist':
                    {'lb': -10,
                    'ub': 10,
                    'scaler': 1.},
                'PropellerSlipstreamWingModel.DESIGNVARIABLES.rotor_0_twist':
                    {'lb': 0,
                    'ub': 90,
                    'scaler': 1./45},
                'PropellerSlipstreamWingModel.DESIGNVARIABLES.rotor_1_twist':
                    {'lb': 0,
                    'ub': 90,
                    'scaler': 1./45}
                }

constraints = {'PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.aero_funcs.CL':
                    {'equals': 0.08104193},
                'PropellerSlipstreamWingModel.HELIX_COUPLED.thrust_total':
                    {'equals': 10.},
                'PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.total_perf.D':
                    {'equals': 20.}
                }

class WingSlipstreamPropAnalysis(om.Group):
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo)

    def setup(self):
        self.add_subsystem('PropellerSlipstreamWingModel',
                           subsys=WingSlipstreamProp(WingPropInfo=PROWIM_wingpropinfo))

    def configure(self):
        # === Add design variables ===
        for design_var_key in design_vars.keys():
            self.add_design_var(design_var_key,
                                lower=design_vars[design_var_key]['lb'],
                                upper=design_vars[design_var_key]['ub'], 
                                scaler=design_vars[design_var_key]['scaler'])
        
        # === Add constraints ===
        for constraints_key in constraints.keys():
            self.add_constraint(constraints_key,
                                equals=constraints[constraints_key]['equals'])

        # === Add objective ===
        for objective_key in objective.keys():
            self.add_objective(objective_key,
                            scaler=objective[objective_key]['scaler'])


if __name__ == '__main__':
    prob = om.Problem()
    prob.model = WingSlipstreamPropAnalysis(WingPropInfo=PROWIM_wingpropinfo)

    # === Analysis ===
    prob.setup()
    prob.run_model()
    
    partials = prob.check_partials(compact_print=True, show_only_incorrect=True, 
                                   includes=['*PropellerSlipstreamWingModel.RETHORST*'], 
                                   form='central', step=1e-8)

    print_results(design_vars=design_vars, constraints=constraints, objective=objective,
                  prob=prob, kind="Initial Analysis")
        
    Cl_orig = prob['PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.aero_funcs.liftcoeff.Cl'].copy()
    quit()
    # === Optimisation ===
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.opt_settings = {
        "Major feasibility tolerance": 1.0e-5,
        "Major optimality tolerance": 1.0e-5,
        "Minor feasibility tolerance": 1.0e-5,
        "Verify level": -1,
        "Function precision": 1.0e-6,
        "Nonderivative linesearch": None,
        "Print file": os.path.join(BASE_DIR, 'results', "opt_SNOPT_print.txt"),
        "Summary file": os.path.join(BASE_DIR, 'results', "opt_SNOPT_summary.txt")
    }
    
    print('==========================================================')
    print('====================== Optimisation ======================')
    print('==========================================================')
    prob.setup()
    prob.run_driver()

    Cl_opt = prob['PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.aero_funcs.liftcoeff.Cl']

    print_results(design_vars=design_vars, constraints=constraints, objective=objective,
                  prob=prob, kind="Results")
    
    # === Plotting ===
    savepath = os.path.join(BASE_DIR, 'results', 'optimisation_results.png')
    
    plot_CL(BASE_DIR=BASE_DIR, span=PROWIM_wingpropinfo.wing.span, Cl_opt=Cl_opt, Cl_orig=Cl_orig, savepath=savepath)
