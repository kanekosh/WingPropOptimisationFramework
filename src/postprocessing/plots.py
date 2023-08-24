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
    
    # === Misc variables ===
    span = wingpropinfo.wing.span
    
    # === Objective, constraints and DVs ===
        # Original
    first_case = database.get_cases()[0]
    design_variables_orig = first_case.get_design_vars()
    constraints_orig = first_case.get_constraints()
    objective_orig = first_case.get_objectives()
    
        # Optimised
    last_case = database.get_cases()[-1]
    design_variables_opt = last_case.get_design_vars()
    constraints_opt = last_case.get_constraints()
    objective_opt = last_case.get_objectives()
    
    # === Misc variables ===
    try:
        CL_orig = first_case['PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.Cl']
        CL_opt = last_case['PropellerSlipstreamWingModel.OPENAEROSTRUCT.AS_point_0.wing_perf.Cl']

        # === Plotting misc variables ===
        var_x = np.linspace(-span/2, span/2, len(CL_orig))
        optimisation_result_plot(design_variable_array=var_x, original=CL_orig, optimised=CL_opt,
                                    label=r"$C_L$", xlabel=r'Wing spanwise location', ylabel=r"$C_L$",
                                    savepath=os.path.join(savedir, f'CL_Wing'))
    except Exception as e:
        print(f'No CL found: {e}')
        
    try:
        for misckey in first_case.outputs.keys():
            if 'velocity_distribution' in misckey:
                veldistr_orig = first_case.outputs[misckey]
                veldistr_opt = last_case.outputs[misckey]

                # === Plotting misc variables ===
                var_x = np.linspace(0, 1, len(veldistr_orig))
                optimisation_result_plot(design_variable_array=var_x, original=veldistr_orig, optimised=veldistr_opt,
                                            label=r"$V$", xlabel=r'Prop blade location', ylabel=r"$V$",
                                            savepath=os.path.join(savedir, f'vel_distr_prop'))
    except Exception as e:
        print(f'No CL found: {e}')
    
    # === Plotting design variables ===
    for dv_key in design_variables_orig.keys():
        variable_orig = design_variables_orig[dv_key]
        variable_opt = design_variables_opt[dv_key]
        
        str_lst = dv_key.split(".")
        var_name = str_lst[-1]
        
        if 'rotor' in var_name or 'geodef_parametric' in var_name:
            # Propeller Plotting
            variable = var_name[8:]
            var_x = np.linspace(0, 1, len(variable_orig))
            optimisation_result_plot(design_variable_array=var_x, original=variable_orig, optimised=variable_opt,
                label=var_name, xlabel=r'Propeller spanwise location $y$', ylabel=var_name,
                savepath=os.path.join(savedir, f'Prop_{variable}'))
        
        else:
            # Wing Plotting
            var_x = np.linspace(-span/2, span/2, len(variable_orig))
            optimisation_result_plot(design_variable_array=var_x, original=variable_orig, optimised=variable_opt,
                label=var_name, xlabel=r'Wing spanwise location', ylabel=var_name,
                savepath=os.path.join(savedir, f'Wing_{var_name}'))
            
    # === Plotting objectives ===
    for dv_key in objective_orig.keys():
        variable_orig = objective_orig[dv_key]
        variable_opt = objective_opt[dv_key]
        
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
            var_x = np.linspace(-span/2, span/2, len(variable_orig))
            optimisation_result_plot(design_variable_array=var_x, original=variable_orig, optimised=variable_opt,
                label=var_name, xlabel=r'Wing spanwise location', ylabel=var_name,
                savepath=os.path.join(savedir, f'Wing_{var_name}'))
    
    # === Plotting constraints ===
    for dv_key in constraints_opt.keys():
        variable_orig = constraints_opt[dv_key]
        variable_opt = constraints_orig[dv_key]
        
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
            var_x = np.linspace(-span/2, span/2, len(variable_orig))
            optimisation_result_plot(design_variable_array=var_x, original=variable_orig, optimised=variable_opt,
                label=var_name, xlabel=r'Wing spanwise location', ylabel=var_name,
                savepath=os.path.join(savedir, f'Wing_{var_name}'))

def plot_cons_objs():
    ...

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


# if __name__=='__main__':
#     all_plots(db_name='/home/mdolabuser/mount/code/framework/WingPropOptimisationFramework/examples/optimisation/results/data.db',
#               wingpropinfo=PROWIM_wingpropinfo,
#               savedir='.')