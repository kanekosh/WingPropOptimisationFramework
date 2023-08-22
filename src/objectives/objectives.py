# --- Built-ins ---

# --- Internal ---

# --- External ---
import openmdao.api as om


class ObjectivePower(om.ExplicitComponent):
    def initialize(self):
        ...

    def setup(self):
        ...
        
    def compute(self, inputs, outputs):
        ...
        
    def compute_partials(self, inputs, partials):
        ...