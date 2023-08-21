# --- Built-ins ---
from pathlib import Path
import os
import logging
import copy

# --- Internal ---
from src.base import ParamInfo, WingPropInfo, WingInfo, PropInfo, AirfoilInfo
from src.tools.tools import print_results
from src.integration.model_coupling import WingSlipstreamProp
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

# --- External ---
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots


logging.getLogger('matplotlib.font_manager').disabled = True
BASE_DIR = Path(__file__).parents[0]

objective = 'PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.total_perf.CD'
design_vars = {'PropellerSlipstreamWingModel.DESIGNVARIABLES.twist':
                    {'lb': -10,
                    'ub': 10,
                    'scaler': 1.}
                }

constraints = {'PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.aero_funcs.CL':
                    {'equals': 0.08104193}
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
        self.add_objective(objective,
                           scaler=1000)


if __name__ == '__main__':
    prob = om.Problem()
    prob.model = WingSlipstreamPropAnalysis(WingPropInfo=PROWIM_wingpropinfo)

    print('==========================================================')
    print('==================== Initial Analysis ====================')
    print('==========================================================')

    prob.setup()
    prob.run_model()
    
    print(objective, ' :  ', prob[objective])
    
    for design_var_key in design_vars.keys():
        print(design_var_key, ' : ', prob[design_var_key])
        
    for constraint_key in constraints.keys():
        print(constraint_key, ' : ', prob[constraint_key])
        
    Cl_orig = prob['PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.aero_funcs.liftcoeff.Cl'].copy()
    
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
    
    print('==========================================================')
    print('======================= Results ==========================')
    print('==========================================================')
    print(objective, ' :  ', prob[objective])
    
    for design_var_key in design_vars.keys():
        print(design_var_key, ' : ', prob[design_var_key])
        
    for constraint_key in constraints.keys():
        print(constraint_key, ' : ', prob[constraint_key])

    plt.style.use(niceplots.get_style())
    _, ax = plt.subplots(figsize=(10, 7))

    spanwise = np.linspace(-PROWIM_wingpropinfo.wing.span/2,
                           PROWIM_wingpropinfo.wing.span/2,
                           len(Cl_opt))
    ax.plot(spanwise, Cl_orig, label='Lift coefficient, original')
    ax.plot(spanwise, Cl_opt, label='Lift coefficient, optimised')

    ax.set_xlabel(r'Spanwise location $y$')
    ax.set_ylabel(r'$C_L\cdot c$')
    ax.legend()
    niceplots.adjust_spines(ax, outward=True)

    plt.savefig(os.path.join(BASE_DIR, 'results', 'optimisation_results.png'))
