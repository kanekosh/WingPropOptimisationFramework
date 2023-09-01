__author__ = "Bernardo Pacini"
__email__ = "bpacini@umich.edu"
__date__ = "Aug. 1st, 2023 Tue"
__status__ = "Production"

import openmdao.api as om
import numpy as np
import pyspline


class thrust_drag(om.ExplicitComponent):
    """
    This class computes the constraint value for a thrust / drag constraint
    """

    def initialize(self):
        self.options.declare("drag_offset", recordable=False)

    def setup(self):
        self.drag_offset = self.options["drag_offset"]

        # Thrust Input
        self.add_input("thrust", shape_by_conn=True)

        # Drag Input
        self.add_input("drag", shape=1)

        # Residual Output
        self.add_output("thrust_drag", shape=1)

    def compute(self, inputs, outputs):
        thrust_total = inputs["thrust"][0, 0]
        outputs["thrust_drag"] = -thrust_total - (inputs["drag"] + self.drag_offset)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # Forward Mode
        if mode == "fwd":
            if "thrust" in d_inputs:
                if "thrust_drag" in d_outputs:
                    d_outputs["thrust_drag"] += -d_inputs["thrust"][0, 0]
            if "drag" in d_inputs:
                if "thrust_drag" in d_outputs:
                    d_outputs["thrust_drag"] -= d_inputs["drag"]

        # Reverse Mode
        elif mode == "rev":
            if "thrust_drag" in d_outputs:
                if "thrust" in d_inputs:
                    d_inputs["thrust"][0, 0] += -d_outputs["thrust_drag"]
            if "thrust_drag" in d_outputs:
                if "drag" in d_inputs:
                    d_inputs["drag"] -= d_outputs["thrust_drag"]


class bspline_interpolant(om.ExplicitComponent):
    """
    This class computes a spline interpolant
    """

    def initialize(self):
        self.options.declare("s", recordable=False)
        self.options.declare("x", recordable=False)
        self.options.declare("order", recordable=False)
        self.options.declare("deriv_1", recordable=False)
        self.options.declare("deriv_2", recordable=False)

    def setup(self):
        self.s = self.options["s"]
        self.x = self.options["x"]
        self.order = self.options["order"]
        self.deriv_1 = self.options["deriv_1"]
        self.deriv_2 = self.options["deriv_2"]

        self.n_ctl_pts = np.size(self.s)
        self.n_eval_pts = np.size(self.x)

        # Add Control Points Inputs
        self.add_input("ctl_pts", shape=self.n_ctl_pts)

        # Add Evaluation Point Output
        self.add_output("y", shape=np.size(self.x))
        if self.deriv_1:
            self.add_output("dy", shape=self.n_eval_pts)
        if self.deriv_2:
            self.add_output("d2y", shape=self.n_eval_pts)

        # Initialize Spline
        self.curve = pyspline.Curve(s=self.s, x=np.ones(self.n_ctl_pts), k=self.order)

    def compute(self, inputs, outputs):
        ctl_pts = inputs["ctl_pts"]
        self.curve.coef[:, 0] = ctl_pts[:]

        y = np.zeros(self.n_eval_pts)
        dy = np.zeros(self.n_eval_pts)
        d2y = np.zeros(self.n_eval_pts)

        for i in range(self.n_eval_pts):
            y[i] = self.curve.getValue(self.x[i])
            if self.deriv_1:
                dy[i] = self.curve.getDerivative(self.x[i])
            if self.deriv_2:
                d2y[i] = self.curve.getSecondDerivative(self.x[i])

        outputs["y"] = y
        if self.deriv_1:
            outputs["dy"] = dy
        if self.deriv_2:
            outputs["d2y"] = d2y

    def setup_partials(self):
        self.declare_partials("*", "*")

    def compute_partials(self, inputs, partials):

        y = np.zeros((self.n_eval_pts, self.n_ctl_pts))
        dy = np.zeros((self.n_eval_pts, self.n_ctl_pts))
        d2y = np.zeros((self.n_eval_pts, self.n_ctl_pts))

        for i in range(self.n_eval_pts):
            for j in range(self.n_ctl_pts):
                self.curve.coef[:, 0] = 0.0
                self.curve.coef[j, 0] = 1.0

                y[i, j] = self.curve.getValue(self.x[i])
                if self.deriv_1:
                    dy[i, j] = self.curve.getDerivative(self.x[i])
                if self.deriv_2:
                    d2y[i, j] = self.curve.getSecondDerivative(self.x[i])

        partials["y", "ctl_pts"] = y
        if self.deriv_1:
            partials["dy", "ctl_pts"] = dy
        if self.deriv_2:
            partials["d2y", "ctl_pts"] = d2y


class radius_span(om.ExplicitComponent):
    """
    This class computes span sections based on a total radius design variable
    """

    def initialize(self):
        self.options.declare("n_sec", recordable=False)
        self.options.declare("r_hub", recordable=False)

    def setup(self):
        self.n_sec = self.options["n_sec"]
        self.r_hub = self.options["r_hub"]

        # Radius Input
        self.add_input("radius", shape=1)

        # Residual Output
        self.add_output("span", shape=self.n_sec)

    def compute(self, inputs, outputs):
        outputs["span"] = (inputs["radius"] - self.r_hub) / self.n_sec

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # Forward Mode
        if mode == "fwd":
            if "radius" in d_inputs:
                if "span" in d_outputs:
                    d_outputs["span"] += d_inputs["radius"] / self.n_sec

        # Reverse Mode
        elif mode == "rev":
            if "span" in d_outputs:
                if "radius" in d_inputs:
                    d_inputs["radius"] += np.sum(d_outputs["span"] / self.n_sec)
