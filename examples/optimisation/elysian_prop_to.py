# --- Built-ins ---
from pathlib import Path
import os
import logging
import copy

# --- Internal ---
from src.utils.tools import print_results
from src.postprocessing.plots import all_plots, stackedplots_prop
from src.integration.coupled_groups_optimisation import PropOptimisationPitch
from examples.example_classes.elysian_classes import elysian_wingpropinfo, elysian_prop

# --- External ---
import openmdao.api as om
import numpy as np


logging.getLogger('matplotlib.font_manager').disabled = True
BASE_DIR = Path(__file__).parents[0]

def propopt(vinf: float, thrust_required: float, rot_rate: float, rho: float)->np.array:
    elysian_wingpropinfo.propeller = [elysian_prop]
    elysian_wingpropinfo.nr_props = 1
    elysian_wingpropinfo.propeller[0].rot_rate = rot_rate
    elysian_wingpropinfo.parameters.vinf = vinf
    elysian_wingpropinfo.parameters.air_density = rho
        
    objective = {
                'HELIX_COUPLED.power_total':
                    {'scaler': 1./10628923.21085738}
                }

    design_vars = {
                    'DESIGNVARIABLES.rotor_0_pitch':
                        {'lb': 0,
                        'ub': 90,
                        'scaler': 1./10},
                    'HELIX_0.om_helix.geodef_parametric_0_rot_rate':
                        {'lb': 0,
                        'ub': 3000,
                        'scaler': 1./160.},
                    }

    constraints = {
                    'HELIX_COUPLED.thrust_total':
                        {'scaler': thrust_required}
                    }
    
    prob = om.Problem()
    prob.model = PropOptimisationPitch(  WingPropInfo=elysian_wingpropinfo,
                                        objective=objective,
                                        constraints=constraints,
                                        design_vars=design_vars)
    
    prob.setup()
    prob.run_model()
    
    prob.check_partials(includes=['*HELIX_PITCH*'], compact_print=True, show_only_incorrect=True)
    quit()
    print_results(design_vars=design_vars, constraints=constraints, objective=objective,
                  prob=prob, kind="Initial Analysis")

    # === Optimisation ===
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings = {
    "Major feasibility tolerance": 1.0e-5,
    "Major optimality tolerance": 1.0e-5,
    "Minor feasibility tolerance": 1.0-5,
    "Verify level": -1,
    "Function precision": 1.0e-6,
    # "Major iterations limit": 50,
    "Nonderivative linesearch": None,
    "Print file": os.path.join(BASE_DIR, 'results', 'optimisation_proponly_print.out'),
    "Summary file": os.path.join(BASE_DIR, 'results', 'optimisation_proponly_summary.out')
    }
        
        # Initialise recorder
    db_name = os.path.join(BASE_DIR, 'results', 'data_propeller.db')
    
    recorder = om.SqliteRecorder(db_name)
    prob.driver.add_recorder(recorder)
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['includes'] = [
                                                    "HELIX_0.om_helix.rotorcomp_0_radii",
                                                    "HELIX_0.om_helix.rotorcomp_0_velocity_distribution",
                                                    "blade_chord_spline_0.y"
                                                 ]
    
    print('==========================================================')
    print('====================== Optimisation ======================')
    print('==========================================================')
    prob.setup(mode='rev')
    prob.run_driver()
    
    print_results(design_vars=design_vars, constraints=constraints, objective=objective,
                  prob=prob, kind="Initial Analysis")
    
    return prob["HELIX_COUPLED.power_total"]
    
if __name__=='__main__':
    # === Initiation ===
    vinf = 0.
    dt = 0.001

    # === Constants ===
    g = 9.80665
    rho = 1.225
    nr_props = 8
    
    # === Input variables ===
    mass = 70_000 # kg
    v_decision = 80 # m/s
    distance_runway = 2000 # m
    max_decelaration = 0.5*g # m/s^2
    
    rot_rate = 160
    
    cd = 0.04
    S = 148.1
    
    # === Calculations ===
    t = v_decision/max_decelaration
    distance_stopping = 0.5 * max_decelaration * t**2 # assuming constant decelaration
    distance_acceleration = distance_runway - distance_stopping
    acceleration_average = v_decision**2/(2*distance_acceleration)
    
    thrust_netto = acceleration_average*mass
    
    power = []
    
    while vinf<v_decision:
        drag = 0.5*rho*vinf**2*S*cd
        thrust_required = thrust_netto+drag
        
        power.append(propopt(vinf=vinf, 
                            thrust_required=thrust_required/nr_props, 
                            rot_rate=rot_rate, 
                            rho=rho))
        
        vinf+=acceleration_average*dt