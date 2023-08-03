# --- Built-ins ---
import unittest

# --- Internal ---
from src.base import WingPropInfo, WingInfo, PropInfo
from src.models.propeller_model import PropellerModel
from src.models.wing_model import WingModel
from src.models.slipstream_model import SlipStreamModel

# --- External ---
import numpy as np
import openmdao.api as om

class WingSlipstreamProp(om.Group):
    def initialize(self):
        self.options.declare('WingPropInfo', default=WingPropInfo())
    
    def setup(self):
        # === Options ===
        wingpropinfo = self.options["WingPropInfo"]
        
        # === Components ===
        for propeller_nr in range(wingpropinfo.nr_props):
            self.add_subsystem(f'HELIX_{propeller_nr}', subsys=PropellerModel())
        
        self.add_subsystem('RETHORST', subsys=SlipStreamModel())
        self.add_subsystem('OPENAEROSTRUCT', subsys=WingModel())
        
        # === Explicit connecting ===        
        self.connect("velocity_distribution", "AS_point_0.coupled.aero_states.velocity_distribution")
        self.connect("rethorst_correction", "AS_point_0.coupled.aero_states.rethorst_correction")