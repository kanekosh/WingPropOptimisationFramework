# --- Built-ins ---
import os

# --- Internal ---
from src.base import WingPropInfo
from src.integration.master_model import WingSlipstreamPropOptimisation
from src.utils.tools import print_results
from src.postprocessing.plots import all_plots

# --- External ---
import matplotlib.pyplot as plt
import niceplots
import numpy as np
import openmdao.api as om


class MainWingPropOptimisation:
    def __init__(self, wingpropinfo: WingPropInfo,
                        objective: dict, constraints: dict, design_variables: dict,
                        database_savefile: str, result_dir: str):
        self.wingpropinfo = wingpropinfo
        self.objective = objective
        self.constraints = constraints
        self.design_variables = design_variables

        self.database_savefile = database_savefile
        self.result_dir = result_dir
    
    def __post_init__(self):
        self.prob = om.Problem()
        self.prob.model = WingSlipstreamPropOptimisation(WingPropInfo=self.wingpropinfo,
                                                    objective=self.objective,
                                                    constraints=self.constraints,
                                                    design_vars=self.design_variables)        
       
        # === Optimisation specific setup ===
        self.prob.driver = om.pyOptSparseDriver()
        self.prob.driver.options['optimizer'] = 'SLSQP'
        self.prob.driver.opt_settings = {
            "MAXIT": 10,
            # 'IFILE': os.path.join(BASE_DIR, 'results', 'optimisation_log.out')
        }
        
            # Initialise recorder
        db_name = os.path.join(self.database_savefile)
        savepath = os.path.join(self.result_dir)
        
        recorder = om.SqliteRecorder(db_name)
        self.prob.driver.add_recorder(recorder)
        self.prob.driver.add_recorder(recorder)
        self.prob.driver.recording_options['includes'] = ["PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.Cl"]
        
    def run_optimisation(self):
        var = "Optimisation"
        print('==========================================================')
        print(f'{var:=^60}')
        print('==========================================================')
        
        self.prob.setup()
        self.prob.run_driver()
        
        self.prob.cleanup() # close all recorders
        
        print_results(design_vars=self.design_vars, constraints=self.constraints, objective=self.objective,
                  prob=self.prob, kind="Optimisation")
        
        self.prob.cleanup() # close all recorders
        
    def run_analysis(self):
        var = "Analysis"
        print('==========================================================')
        print(f'{var:=^60}')
        print('==========================================================')
        
        self.prob.setup()
        self.prob.run_model()
        
        print_results(design_vars=self.design_vars, constraints=self.constraints, objective=self.objective,
                  prob=self.prob, kind="Analysis")
        
    def visualise_results(self):
        ...