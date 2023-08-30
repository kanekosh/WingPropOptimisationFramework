# --- Built-ins ---
import os
import logging

# --- Internal ---
from src.base import WingPropInfo
from src.postprocessing.utils import get_niceColors, get_delftColors, get_SuperNiceColors
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

# --- External ---
import matplotlib.pyplot as plt
import matplotlib as mpl
import niceplots
from openmdao.recorders.sqlite_reader import SqliteCaseReader
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logging.getLogger('matplotlib.font_manager').disabled = True


def all_plots(db_name: str,
              wingpropinfo: WingPropInfo,
              savedir: str,
              *kwargs) -> None:
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
    spanwise_mesh = wingpropinfo.vlm_mesh_control_points
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

            elif 'AS_point_0.wing_perf.Cl' in misckey:
                veldistr_orig = first_case.outputs[misckey]
                veldistr_opt = last_case.outputs[misckey]

                # === Plotting misc variables ===
                var_x = np.linspace(-span/2, span/2, len(veldistr_orig))
                optimisation_result_plot(design_variable_array=spanwise_mesh,
                                         original=veldistr_orig,
                                         optimised=veldistr_opt,
                                         label=r"$C_L$", xlabel=r'Wing spanwise location', ylabel=r"$C_L$",
                                         savepath=os.path.join(savedir, f'CL_Wing'))

            elif 'wing.geometry.twist' in misckey:
                twist_orig = first_case.outputs[misckey][0]
                twist_opt = last_case.outputs[misckey][0]

                # === Plotting misc variables ===
                var_x = wingpropinfo.vlm_mesh_control_points
                optimisation_result_plot(design_variable_array=var_x, original=twist_orig, optimised=twist_opt,
                                         label=r"$Twist$", xlabel=r'Wing spanwise location', ylabel=r"$Twist$",
                                         savepath=os.path.join(savedir, f'Wing_twist_DV'))

    except Exception as e:
        print(f'No CL found: {e}')

    # === Plotting design variables ===
    for index, dv_key in enumerate(design_variables_orig.keys()):
        variable_orig = design_variables_orig[dv_key]
        variable_opt = design_variables_opt[dv_key]

        if len(variable_orig) != 1:
            str_lst = dv_key.split(".")
            var_name = str_lst[-1]

            if 'rotor' in var_name or 'geodef_parametric' in var_name:
                # Propeller Plotting
                variable = var_name[8:]
                var_x = np.linspace(0, 1, len(variable_orig))
                optimisation_result_plot(design_variable_array=var_x, original=variable_orig, optimised=variable_opt,
                                         label=var_name, xlabel=r'Propeller spanwise location $y$', ylabel=var_name,
                                         savepath=os.path.join(savedir, f'Prop_{variable}_{index}'))

            else:
                # Wing Plotting
                var_x = np.linspace(-span/2, span/2, len(variable_orig))
                optimisation_result_plot(design_variable_array=var_x,
                                         original=variable_orig,
                                         optimised=variable_opt,
                                         label=var_name, xlabel=r'Wing spanwise location', ylabel=var_name,
                                         savepath=os.path.join(savedir, f'Wing_{var_name}'))

    # === Plotting objectives ===
    for dv_key in objective_orig.keys():
        variable_orig = objective_orig[dv_key]
        variable_opt = objective_opt[dv_key]

        if len(variable_orig) != 1:
            str_lst = dv_key.split(".")
            var_name = str_lst[-1]

            if var_name[:5] == 'rotor':
                # Propeller Plotting
                variable = var_name[8:]
                var_x = np.linspace(0, 1, len(variable_orig))
                optimisation_result_plot(design_variable_array=var_x, original=variable_orig, optimised=variable_opt,
                                         label=var_name, xlabel=r'Propeller spanwise location $y$', ylabel=var_name,
                                         savepath=os.path.join(savedir, f'Prop_{variable}'))

            else:
                # Wing Plotting
                var_x = np.linspace(-span/2, span/2, len(variable_orig))
                optimisation_result_plot(design_variable_array=var_x,
                                         original=variable_orig,
                                         optimised=variable_opt,
                                         label=var_name, xlabel=r'Wing spanwise location', ylabel=var_name,
                                         savepath=os.path.join(savedir, f'Wing_{var_name}'))

    # === Plotting constraints ===
    for dv_key in constraints_opt.keys():
        variable_orig = constraints_opt[dv_key]
        variable_opt = constraints_orig[dv_key]

        if len(variable_orig) != 1:
            str_lst = dv_key.split(".")
            var_name = str_lst[-1]

            if var_name[:5] == 'rotor':
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
                                         label=var_name, xlabel=r'Wing spanwise location [$m$]', ylabel=rf'${var_name}$',
                                         savepath=os.path.join(savedir, f'Wing_{var_name}'))

    scatter_plots(db_name=db_name,
                  savedir=savedir)


def plot_optimality(SNOPT_output: str):
    ...


def scatter_plots(db_name: str,
                  savedir: str):
    database = SqliteCaseReader(db_name, pre_load=True)

    # === Objective, constraints and DVs ===
    # Original
    cases = database.get_cases()
    for varkey in cases[0].outputs.keys():
        var = []
        if np.size(cases[0][varkey]) == 1:
            for case in cases:
                var.append(case.outputs[varkey])

            design_variable_array = np.linspace(0, len(var), len(var))

            ylabel = varkey.split('.')[-1]

            optimisation_singlevalue_results(design_variable_array=design_variable_array,
                                             xlabel='Iterations', ylabel=ylabel,
                                             savepath=os.path.join(
                                                 savedir, ylabel),
                                             variable=var)


def optimisation_result_plot(design_variable_array: np.array, original: np.array, optimised: np.array,
                             label: str, xlabel: str, ylabel: str,
                             savepath: str) -> None:
    colors = get_niceColors()
    delftcolors = get_delftColors()
    supernicecolors = get_SuperNiceColors()
    margin = 1.015

    plt.style.use(niceplots.get_style())
    _, ax = plt.subplots(figsize=(10, 7))

    spanwise = design_variable_array
    ax.plot(spanwise, original,
            label=f'{label}, original', color=colors['Orange'])
    ax.plot(spanwise, optimised,
            label=f'{label}, optimised', color=colors['Cyan'], linestyle='dashed')

    # ax.plot([0.332-0.12, 0.332-0.12], [0., 0.25], color=delftcolors['Grey'], linewidth=0.5)
    # ax.plot([0.332+0.12, 0.332+0.12], [0., 0.25], color=delftcolors['Grey'], linewidth=0.5)
    # ax.arrow(x=0.332, y=0.02, dx=0.06, dy=0.)

    # ax.text(x=0.332-0.1, y=0.03, s='Propeller', fontdict={'fontsize': mpl.rcParams['axes.titlesize'],
    #                                                         'fontweight': 'ultralight',
    #                                                         'fontsize': 12})

    ax.set_xlabel(xlabel, fontweight='ultralight')
    ax.set_ylabel(ylabel, fontweight='ultralight')

    ax.set_ylim((
        min(min(original), min(optimised))*margin,
        max(max(original), max(optimised))*margin)
    )
    ax.set_xlim((
        min(spanwise)*margin,
        max(spanwise)*margin)
    )
    ax.legend()
    niceplots.adjust_spines(ax, outward=True)

    plt.savefig(savepath)


def optimisation_singlevalue_results(design_variable_array: np.array,
                                     xlabel: str, ylabel: str,
                                     savepath: str,
                                     **kwargs) -> None:
    plt.style.use(niceplots.get_style())
    _, ax = plt.subplots(figsize=(10, 7))

    spanwise = design_variable_array
    for key in kwargs.keys():
        ax.plot(spanwise, kwargs[key])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    niceplots.adjust_spines(ax, outward=True)

    plt.savefig(savepath)


if __name__ == '__main__':
    # scatter_plots(savedir='.',
    #     db_name='/home/mdolabuser/mount/code/framework/WingPropOptimisationFramework/examples/optimisation/results/data.db')
    all_plots(db_name='/home/mdolabuser/mount/code/framework/WingPropOptimisationFramework/examples/optimisation/results/data.db',
              wingpropinfo=PROWIM_wingpropinfo,
              savedir='.')
