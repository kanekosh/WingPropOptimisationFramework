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
    print(objective, ' :  ', prob[objective])
    
    for design_var_key in design_vars.keys():
        print(design_var_key, ' : ', prob[design_var_key])
        
    for constraint_key in constraints.keys():
        print(constraint_key, ' : ', prob[constraint_key])
        
def plot_CL(BASE_DIR: str, span: float, 
            Cl_opt: np.array, Cl_orig: np.array,
            savepath: str)->None:
    
    plt.style.use(niceplots.get_style())
    _, ax = plt.subplots(figsize=(10, 7))

    spanwise = np.linspace(-span/2,
                           span/2,
                           len(Cl_opt))
    ax.plot(spanwise, Cl_orig, label='Lift coefficient, original')
    ax.plot(spanwise, Cl_opt, label='Lift coefficient, optimised')

    ax.set_xlabel(r'Spanwise location $y$')
    ax.set_ylabel(r'$C_L\cdot c$')
    ax.legend()
    niceplots.adjust_spines(ax, outward=True)

    plt.savefig(savepath)