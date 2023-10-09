# --- Built-ins ---

# --- Internal ---

# --- External ---
import openmdao.api as om
import numpy as np

class PROP_WEIGHTMODEL(om.ExplicitComponent):
    def initialize(self):
        # number of propellers
        self.options.declare("propeller_quantity", default=0)
        # spanwise discretisation of the propeller blade
        self.options.declare("propeller_discretisation", default=0)
        self.options.declare("propeller_discretisation_BEM", default=0)
        self.options.declare("propeller_tipradii", default=[0])
        self.options.declare("propeller_local_refinement", default=2)

    def setup(self):