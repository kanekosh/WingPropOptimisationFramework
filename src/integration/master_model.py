# --- Built-ins ---

# --- Internal ---
from src.base import WingPropInfo, PropInfo, ParamInfo
from src.integration.model_coupling import WingSlipstreamProp
from src.models.propeller_model import PropellerModel, PropellerCoupled

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
        wingpropinfo = self.options['WingPropInfo']
        self.add_subsystem('PropellerSlipstreamWingModel',
                           subsys=WingSlipstreamProp(WingPropInfo=wingpropinfo))

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
            

class PropOptimisation(om.Group):
    def initialize(self):
        self.options.declare('PropInfo', default=PropInfo)
        self.options.declare('ParamInfo', default=ParamInfo)
        self.options.declare('WingPropInfo', default=WingPropInfo)
        self.options.declare('objective', default=dict)
        self.options.declare('constraints', default=dict)
        self.options.declare('design_vars', default=dict)

    def setup(self):
        propinfo = self.options['PropInfo']
        paraminfo = self.options['ParamInfo']
        wingpropinfo = self.options['WingPropInfo']
        
        self.add_subsystem('PropellerModel',
                           subsys=PropellerModel(PropInfo=propinfo,
                                                 ParamInfo=paraminfo))
        
        self.add_subsystem('HELIX_COUPLED', 
                           subsys=PropellerCoupled(WingPropInfo=wingpropinfo))
        
        # HELIX to HELIX_COUPLED
        for index in range(wingpropinfo.nr_props):
            self.connect(f"PropellerModel.om_helix.rotorcomp_0_thrust",
                        f"HELIX_COUPLED.thrust_prop_{index}")
            self.connect(f"PropellerModel.om_helix.rotorcomp_0_power",
                        f"HELIX_COUPLED.power_prop_{index}")

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