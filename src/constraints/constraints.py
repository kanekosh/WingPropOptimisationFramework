# --- Built-ins ---

# --- Internal ---

# --- External ---
import openmdao.api as om


class ConstraintsThrustDrag(om.ExplicitComponent):
    def initialize(self):
        ...

    def setup(self):
        # === Options ===
        
        # === Inputs ===
        self.add_input('thrust_total', shape=1)
        self.add_input('drag_total', shape=1)
        
        # === Inputs ===
        self.add_output('thrust_equals_drag', shape=1)
        
        # === Partials ===
        self.declare_partials('thrust_equals_drag',
                              'thrust_total')
        self.declare_partials('thrust_equals_drag',
                              'drag_total')
        
    def compute(self, inputs, outputs):
        # === Options ===
        
        # === Inputs ===
        thrust_total = inputs['thrust_total']
        drag_total = inputs['drag_total']
        
        outputs['thrust_equals_drag'] = 1-thrust_total/drag_total
        
    def compute_partials(self, inputs, partials):
        # === Options ===
        
        # === Inputs ===
        thrust_total = inputs['thrust_total']
        drag_total = inputs['drag_total']
        
        partials['thrust_equals_drag',
                'thrust_total'] = -1/drag_total
        partials['thrust_equals_drag',
                'drag_total'] = thrust_total/drag_total**2