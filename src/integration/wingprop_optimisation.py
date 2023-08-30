# --- Built-ins ---
import os

# --- Internal ---
from src.base import WingPropInfo
from src.integration.coupled_groups_optimisation import WingSlipstreamPropOptimisation
from src.utils.tools import print_results
from src.postprocessing.plots import all_plots

# --- External ---
import numpy as np
import openmdao.api as om


class MainWingPropOptimisation:
    def __init__(self, wingpropinfo: WingPropInfo,
                        objective: dict, constraints: dict, design_variables: dict,
                        result_dir: str, database_savefile: str,
                        optimizer: str='pyoptsparse', algorithm: str='SNOPT'):
        self.wingpropinfo: WingPropInfo = wingpropinfo
        self.objective: dict = objective
        self.constraints: dict = constraints
        self.design_variables: dict = design_variables

        self.database_savefile: str = database_savefile
        self.results_dir: str = result_dir
        
        self.optimizer: str = optimizer
        self.algorithm: str = algorithm
    
    def __post_init__(self):
        self.prob = om.Problem()
        self.prob.model = WingSlipstreamPropOptimisation(WingPropInfo=self.wingpropinfo,
                                                            objective=self.objective,
                                                            constraints=self.constraints,
                                                            design_vars=self.design_variables)        
       
        if self.optimizer=='pyoptsparse':
            # === Optimisation specific setup ===
            self.prob.driver = om.pyOptSparseDriver()
            self.prob.driver.options['optimizer'] = self.algorithm
            self.prob.driver.opt_settings = {
                "Major feasibility tolerance": 1.0e-8,
                "Major optimality tolerance": 1.0e-8,
                "Minor feasibility tolerance": 1.0e-8,
                "Verify level": -1,
                "Function precision": 1.0e-6,
                # "Major iterations limit": 50,
                "Nonderivative linesearch": None,
                "Print file": os.path.join(self.results_dir, 'optimisation_print_wingprop.out'),
                "Summary file": os.path.join(self.results_dir, 'optimisation_summary_wingprop.out')
            }
        
            # Initialise recorder
        self.db_name = os.path.join(self.results_dir, self.database_savefile)
        
        recorder = om.SqliteRecorder(self.db_name)
        self.prob.driver.add_recorder(recorder)
        self.prob.driver.add_recorder(recorder)
        self.prob.driver.recording_options['includes'] = [  "OPENAEROSTRUCT.wing.geometry.twist",
                                                            "OPENAEROSTRUCT.AS_point_0.wing_perf.Cl",
                                                            "OPENAEROSTRUCT.AS_point_0.wing_perf.CDi",
                                                            'OPENAEROSTRUCT.AS_point_0.total_perf.L',
                                                            'OPENAEROSTRUCT.AS_point_0.total_perf.D',
                                                            'OPENAEROSTRUCT.AS_point_0.total_perf.D',
                                                            'RETHORST.velocity_distribution']
        
    def run_optimisation(self):
        var = "Optimisation"
        print('==========================================================')
        print(f'{var:=^60}')
        print('==========================================================')
        
        self.prob.setup()
        self.prob.run_driver()
        
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
        
        self.prob.cleanup() # close all recorders
        
    def visualise_results(self):
        all_plots(db_name=self.db_name,
                    wingpropinfo=self.wingpropinfo,
                    savedir=self.results_dir)