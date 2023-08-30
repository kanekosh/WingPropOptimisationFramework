# --- Built-ins ---
import unittest

# --- Internal ---
from src.base import WingPropInfo, PropInfo, ParamInfo
from src.models.propeller_model import PropellerModel, PropellerCoupled
from src.models.wing_model import WingModel
from src.models.slipstream_model import SlipStreamModel
from src.models.parameters import Parameters
from src.models.design_variables import DesignVariables
from src.constraints.constraints import ConstraintsThrustDrag

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
        # === Options ===
        wingpropinfo = self.options["WingPropInfo"]

        # === Components ===
            # Inputs
        self.add_subsystem('PARAMETERS', subsys=Parameters(
            WingPropInfo=wingpropinfo))

        self.add_subsystem('DESIGNVARIABLES', subsys=DesignVariables(
            WingPropInfo=wingpropinfo))

            # Modules
        for propeller_nr in range(wingpropinfo.nr_props):
            self.add_subsystem(f'HELIX_{propeller_nr}',
                               subsys=PropellerModel(ParamInfo=wingpropinfo.parameters,
                                                     PropInfo=wingpropinfo.propeller[propeller_nr]))
        
        self.add_subsystem('RETHORST',
                           subsys=SlipStreamModel(WingPropInfo=wingpropinfo))

        self.add_subsystem('OPENAEROSTRUCT',
                           subsys=WingModel(WingPropInfo=wingpropinfo))
        
        self.add_subsystem('HELIX_COUPLED', 
                           subsys=PropellerCoupled(WingPropInfo=wingpropinfo))

            # Outputs
        self.add_subsystem('CONSTRAINTS', 
                           subsys=ConstraintsThrustDrag())
        
        # === Explicit connections ===
        # PARAMS to HELIX
        for index, _ in enumerate(wingpropinfo.propeller):
            self.connect(f"PARAMETERS.rotor_{index}_radius",
                         f"HELIX_{index}.om_helix.geodef_parametric_0_span")

        # PARAMS to RETHORST
        self.connect("PARAMETERS.vinf",
                     "RETHORST.interpolation.vinf")

        self.connect("PARAMETERS.propeller_locations",
                     "RETHORST.propeller_locations")
        self.connect("PARAMETERS.vinf",
                     "RETHORST.correction.vinf")
        self.connect("PARAMETERS.wing_mesh",
                     "RETHORST.wing_mesh")
        self.connect("PARAMETERS.wing_mesh_control_points",
                     "RETHORST.wing_mesh_control_points")

        # PARAMS to OPENAEROSTRUCT
        self.connect('PARAMETERS.vinf',
                     'OPENAEROSTRUCT.v')
        self.connect('PARAMETERS.alpha',
                     'OPENAEROSTRUCT.alpha')
        self.connect('PARAMETERS.Mach_number',
                     'OPENAEROSTRUCT.Mach_number')
        self.connect('PARAMETERS.re',
                     'OPENAEROSTRUCT.re')
        self.connect('PARAMETERS.rho',
                     'OPENAEROSTRUCT.rho')
        self.connect('PARAMETERS.CT',
                     'OPENAEROSTRUCT.CT')
        self.connect('PARAMETERS.R',
                     'OPENAEROSTRUCT.R')
        self.connect('PARAMETERS.W0',
                     'OPENAEROSTRUCT.W0')
        self.connect('PARAMETERS.speed_of_sound',
                     'OPENAEROSTRUCT.speed_of_sound')
        self.connect('PARAMETERS.load_factor',
                     'OPENAEROSTRUCT.load_factor')
        self.connect('PARAMETERS.empty_cg',
                     'OPENAEROSTRUCT.empty_cg')

        # DVs to HELIX
        for index, _ in enumerate(wingpropinfo.propeller):
            self.connect(f"DESIGNVARIABLES.rotor_{index}_twist",
                         f"HELIX_{index}.om_helix.geodef_parametric_0_twist")
            self.connect(f"DESIGNVARIABLES.rotor_{index}_chord",
                         f"HELIX_{index}.om_helix.geodef_parametric_0_chord")
            self.connect(f"DESIGNVARIABLES.rotor_{index}_rot_rate",
                         f"HELIX_{index}.om_helix.geodef_parametric_0_rot_rate")

        # DVs to OPENAEROSTRUCT
        self.connect('DESIGNVARIABLES.twist',
                     'OPENAEROSTRUCT.wing.twist_cp')
        self.connect('DESIGNVARIABLES.chord',
                     'OPENAEROSTRUCT.wing.geometry.chord_cp')
        self.connect('DESIGNVARIABLES.span',
                     'OPENAEROSTRUCT.wing.geometry.span')

        # HELIX to RETHORST
        for index in range(wingpropinfo.nr_props):
            self.connect(f"HELIX_{index}.om_helix.rotorcomp_0_radii",
                        f"RETHORST.interpolation.propeller_radii_BEM_rotor{index}")
            self.connect(f"HELIX_{index}.om_helix.rotorcomp_0_velocity_distribution",
                        f"RETHORST.interpolation.propeller_velocity_BEM_rotor{index}")
            
        # HELIX to HELIX_COUPLED
        for index in range(wingpropinfo.nr_props):
            self.connect(f"HELIX_{index}.om_helix.rotorcomp_0_thrust",
                        f"HELIX_COUPLED.thrust_prop_{index}")
            self.connect(f"HELIX_{index}.om_helix.rotorcomp_0_power",
                        f"HELIX_COUPLED.power_prop_{index}")

        # RETHORST to OPENAEROSTRUCT
        self.connect("RETHORST.velocity_distribution",
                     "OPENAEROSTRUCT.AS_point_0.coupled.aero_states.velocity_distribution")
        self.connect("RETHORST.correction_matrix",
                     "OPENAEROSTRUCT.AS_point_0.coupled.aero_states.rethorst_correction")
        
        # OPENAEROSTRUCT to CONSTRAINTS
        self.connect('OPENAEROSTRUCT.AS_point_0.total_perf.D',
                     'CONSTRAINTS.drag_total')
        
        # HELIX_COUPLED to CONSTRAINTS
        self.connect('HELIX_COUPLED.thrust_total',
                     'CONSTRAINTS.thrust_total')
        
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

class WingOptimisation(om.Group):
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo)
        self.options.declare('objective', default=dict)
        self.options.declare('constraints', default=dict)
        self.options.declare('design_vars', default=dict)

    def setup(self):
        # === Options ===
        wingpropinfo = self.options['WingPropInfo']
        
        # === Add subsystems ===
        self.add_subsystem('DESIGNVARIABLES', subsys=DesignVariables(
            WingPropInfo=wingpropinfo))

        self.add_subsystem('PARAMETERS', subsys=Parameters(
            WingPropInfo=wingpropinfo))

        self.add_subsystem('OPENAEROSTRUCT',
                           subsys=WingModel(WingPropInfo=wingpropinfo))

        # === Connections ===
        # DVs to OPENAEROSTRUCT
        self.connect('DESIGNVARIABLES.twist',
                     'OPENAEROSTRUCT.wing.twist_cp')
        self.connect('DESIGNVARIABLES.chord',
                     'OPENAEROSTRUCT.wing.geometry.chord_cp')
        self.connect('DESIGNVARIABLES.span',
                     'OPENAEROSTRUCT.wing.geometry.span')

        
        # PARAMETERS to OPENAEROSTRUCT
        self.connect('PARAMETERS.vinf',
                     'OPENAEROSTRUCT.v')
        self.connect('PARAMETERS.velocity_distribution',
                     'OPENAEROSTRUCT.AS_point_0.coupled.aero_states.velocity_distribution')
        self.connect('PARAMETERS.alpha',
                     'OPENAEROSTRUCT.alpha')
        self.connect('PARAMETERS.Mach_number',
                     'OPENAEROSTRUCT.Mach_number')
        self.connect('PARAMETERS.re',
                     'OPENAEROSTRUCT.re')
        self.connect('PARAMETERS.rho',
                     'OPENAEROSTRUCT.rho')
        self.connect('PARAMETERS.CT',
                     'OPENAEROSTRUCT.CT')
        self.connect('PARAMETERS.R',
                     'OPENAEROSTRUCT.R')
        self.connect('PARAMETERS.W0',
                     'OPENAEROSTRUCT.W0')
        self.connect('PARAMETERS.speed_of_sound',
                     'OPENAEROSTRUCT.speed_of_sound')
        self.connect('PARAMETERS.load_factor',
                     'OPENAEROSTRUCT.load_factor')
        self.connect('PARAMETERS.empty_cg',
                     'OPENAEROSTRUCT.empty_cg')

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
                    break # We break because otherwise it will loop over the subkey
                        # TODO: horrendous coding convention, there's a better way to solve this issue


        # === Add objective ===
        for objective_key in objective.keys():
            self.add_objective(objective_key,
                            scaler=objective[objective_key]['scaler'])

# TODO: this class seems superfluous, correct?
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

