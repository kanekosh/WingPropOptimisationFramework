# --- Built-ins ---
from pathlib import Path
import os
import logging
import copy

# --- Internal ---
from src.base import WingPropInfo
from src.utils.tools import print_results
from src.postprocessing.plots import all_plots
from src.models.wing_model import WingModel
from src.integration.master_model import WingSlipstreamPropOptimisation
from src.integration.wingprop_optimisation import MainWingPropOptimisation
from examples.example_classes.PROWIM_classes import PROWIM_wingpropinfo

# --- External ---
import openmdao.api as om
import numpy as np


logging.getLogger('matplotlib.font_manager').disabled = True
BASE_DIR = Path(__file__).parents[0]

if __name__ == '__main__':