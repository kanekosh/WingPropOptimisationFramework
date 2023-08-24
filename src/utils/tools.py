# --- Built-ins ---
import unittest

# --- Internal ---

# --- External ---
import openmdao.api as om
import niceplots
import matplotlib.pyplot as plt
import numpy as np

def print_results(design_vars: dict, constraints: dict, objective: dict,
                  prob: om.Problem, kind: str)->None:
    print('============================================================')
    print(f'{kind:=^60}')
    print('============================================================')
    
    for objective_key in objective.keys():
        print(objective_key, ' :  ', prob[objective_key])
    
    for design_var_key in design_vars.keys():
        print(design_var_key, ' : ', prob[design_var_key])
        
    for constraint_key in constraints.keys():
        print(constraint_key, ' : ', prob[constraint_key])
