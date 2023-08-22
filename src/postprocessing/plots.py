# --- Built-ins ---
import os

# --- Internal ---
from src.base import WingPropInfo
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

# --- External ---
import matplotlib.pyplot as plt
import niceplots
from openmdao.recorders.sqlite_reader import SqliteCaseReader
import numpy as np


def all_plots(db_name: str,
              wingpropinfo: WingPropInfo,
              savedir: str,
              *kwargs)->None:
    database = SqliteCaseReader(db_name, pre_load=True)
    
    first_case = database.get_cases()[0]
    design_variables_orig = first_case.get_design_vars()
    constraints_orig = first_case.get_constraints()
    objective_orig = first_case.get_objectives()
    
    last_case = database.get_cases()[-1]
    design_variables_opt = last_case.get_design_vars()
    constraints_opt = last_case.get_constraints()
    objective_opt = last_case.get_objectives()
    
    # === Plotting design variables ===
    for dv_key in design_variables_orig.keys():
        variable_orig = design_variables_orig[dv_key]
        variable_opt = design_variables_opt[dv_key]
        
        str_lst = dv_key.split(".")
        var_name = str_lst[-1]
        
        if var_name[:5]=='rotor':
            # Propeller Plotting
            variable = var_name[8:]
            var_x = np.linspace(0, 1, len(variable_orig))
            optimisation_result_plot(design_variable_array=var_x, original=variable_orig, optimised=variable_opt,
                label=var_name, xlabel=r'Propeller spanwise location $y$', ylabel=var_name,
                savepath=os.path.join(savedir, f'Prop_{variable}'))
        
        else:
            # Wing Plotting
            span = wingpropinfo.wing.span
            var_x = np.linspace(-span/2, span/2, len(variable_orig))
            optimisation_result_plot(design_variable_array=var_x, original=variable_orig, optimised=variable_opt,
                label=var_name, xlabel=r'Wing spanwise location', ylabel=var_name,
                savepath=os.path.join(savedir, f'Wing_{var_name}'))

def optimisation_result_plot(design_variable_array: np.array, original: np.array, optimised: np.array, 
                            label: str, xlabel: str, ylabel: str,
                            savepath: str)->None:
    plt.style.use(niceplots.get_style())
    _, ax = plt.subplots(figsize=(10, 7))

    spanwise = design_variable_array
    ax.plot(spanwise, original, label=f'{label}, original')
    ax.plot(spanwise, optimised, label=f'{label}, optimised')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    niceplots.adjust_spines(ax, outward=True)

    plt.savefig(savepath)
    
def wing_plots():
    ...

if __name__=='__main__':
    all_plots(db_name='/home/mdolabuser/mount/code/framework/WingPropOptimisationFramework/examples/optimisation/results/data.db',
              wingpropinfo=PROWIM_wingpropinfo,
              savedir='.')