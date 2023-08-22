# --- Built-ins ---

# --- Internal ---
from src.base import WingPropInfo

# --- External ---
import matplotlib.pyplot as plt
import niceplots
from openmdao.recorders.sqlite_reader import SqliteCaseReader
import numpy as np


def all_plots(db_name: str, nr_design_variables: int,
              wingpropinfo: WingPropInfo)->None:
    cr = case_reader = SqliteCaseReader(db_name, pre_load=True)
    last_case = next(reversed(cr.get_cases("driver")))
    
    first_case = cr.get_cases()[0]
    first_case_vars = first_case.outputs
    
    design_variables, output_variables = {}, {}
    
    for index, var_key in enumerate(first_case_vars.keys()):
        if index<nr_design_variables:
            design_variables[var_key] = first_case_vars[var_key]
        else:
            output_variables[var_key] = first_case_vars[var_key]
        
    # === Lift distribution ===
    span = wingpropinfo.wing.span
    span_distribution
    optimisation_result_plot(xlabel=-r'Spanwise location $y$', ylabel=r'$C_L\cdot c$')

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

def plot_coefficient(span: float, 
                    coefficient_optimised: np.array, Cl_orig: np.array,
                    savepath: str)->None:
    
    plt.style.use(niceplots.get_style())
    _, ax = plt.subplots(figsize=(10, 7))

    spanwise = np.linspace(-span/2,
                           span/2,
                           len(coefficient_optimised))
    ax.plot(spanwise, Cl_orig, label='Lift coefficient, original')
    ax.plot(spanwise, coefficient_optimised, label='Lift coefficient, optimised')

    ax.set_xlabel(r'Spanwise location $y$')
    ax.set_ylabel(r'$C_L\cdot c$')
    ax.legend()
    niceplots.adjust_spines(ax, outward=True)

    plt.savefig(savepath)

if __name__=='__main__':
    all_plots(db_name='/home/mdolabuser/mount/code/framework/WingPropOptimisationFramework/examples/optimisation/results/data_sample.db',
              nr_design_variables=2,
              nr_outputs=2)