# --- Built-ins ---

# --- Internal ---
from src.base import WingPropInfo
from src.integration.model_coupling import WingSlipstreamProp
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

# --- External ---
import numpy as np
import openmdao.api as om

class WingSlipstreamPropOptimisation(om.Group):
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo)
        self.options.declare('objective', default=dict)
        self.options.declare('constraints', default=dict)
        self.options.declare('design_vars', default=dict)

    def setup(self):
        self.add_subsystem('PropellerSlipstreamWingModel',
                           subsys=WingSlipstreamProp(WingPropInfo=PROWIM_wingpropinfo))

    def configure(self):
        # === Options ===
        objective = self.options['objective']
        constraints = self.options['constraints']
        design_vars = self.options['design_vars']
        
        # === Add design variables ===
        for design_var_key in design_vars.keys():
            self.add_design_var(design_var_key,
                                lower=design_vars[design_var_key]['lb'],
                                upper=design_vars[design_var_key]['ub'], 
                                scaler=design_vars[design_var_key]['scaler'])
        
        # === Add constraints ===
        for constraints_key in constraints.keys():
            for subkey in constraints[constraints_key].keys():
                if len(constraints[constraints_key].keys())==1:
                    if subkey=='equals':
                        self.add_constraint(constraints_key,
                                            equals=constraints[constraints_key]['equals'])
                    elif subkey=='upper':
                        self.add_constraint(constraints_key,
                                            upper=constraints[constraints_key]['upper'])
                    elif subkey=='lower':
                        self.add_constraint(constraints_key,
                                            lower=constraints[constraints_key]['lower'])
                
                elif len(constraints[constraints_key].keys())==2:
                    self.add_constraint(constraints_key,
                                        lower=constraints[constraints_key]['lower'],
                                        upper=constraints[constraints_key]['upper'])
                    break # TODO: there's a better way to solve this issue


        # === Add objective ===
        for objective_key in objective.keys():
            self.add_objective(objective_key,
                            scaler=objective[objective_key]['scaler'])