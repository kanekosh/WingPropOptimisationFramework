# --- Built-ins ---
from dataclasses import dataclass
import copy

# --- Internal ---
from src.utils.meshing import meshing
from openaerostruct.utils.constants import grav_constant

# --- External ---
import numpy as np


@dataclass
class ParamInfo:
    vinf: float
    wing_aoa: float
    mach_number: float
    reynolds_number: float
    speed_of_sound: float
    R: float = 11.165e6
    CT: float = grav_constant * 17.0e-6
    air_density: float = 1.225


@dataclass
class AirfoilInfo:
    label: str
    Cl_alpha: float     # Cl alpha lift slope
    alpha_L0: float     # zero lift angle
    alpha_0: float      # stall angle


@dataclass
class WingInfo:
    label: str
    span: float
    chord: np.array
    twist: np.array
    empty_weight: float
    load_factor: float = 1.
    empty_cg: np.array = np.zeros((3))


@dataclass
class PropInfo:
    label: str
    prop_location: float

    nr_blades: int
    rot_rate: float     # in rad/s
    chord: np.array
    twist: np.array
    span: np.array
    airfoils: list[AirfoilInfo]

    rotation_axis: np.array = np.array([0., 0., 1.])
    hub_offset: float = 0.

    def __post_init__(self):
        assert len(self.chord) == len(self.span)+1, ' Chord should be defined for blade nodes, \
                                                        not control points (length of chord and twist should be one larger than span)'
        assert len(self.twist) == len(self.span)+1, ' Twist should be defined for blade nodes, \
                                                        not control points (length of chord and twist should be one larger than span)'

        self.prop_radius = np.zeros(len(self.span)+1)
        for index, ispan in enumerate(self.span):
            # TODO: this only works for a linearly spaced propeller
            self.prop_radius[index+1] = (index+1)*ispan

    def __str__(self):
        return f'Propeller {self.label}, with {self.nr_blades} blades'


@dataclass
class WingPropInfo:
    nr_props: int
    spanwise_discretisation_wing: int
    spanwise_discretisation_propeller: int
    # TODO: right now it is assumed that all propellers have the same discretisation.
    #           This is not necessarily true
    spanwise_discretisation_propeller_BEM: int

    propeller: list[PropInfo]
    wing: WingInfo
    parameters: ParamInfo

    def __post_init__(self):
        self.spanwise_discretisation_nodes = self.spanwise_discretisation_wing + \
            self.nr_props*self.spanwise_discretisation_propeller + 1

        self.prop_locations = np.zeros((self.nr_props), order='F')
        self.prop_radii = np.zeros(
            (self.nr_props, self.spanwise_discretisation_propeller_BEM+1), order='F')

        # Merge the propeller information into a single array
        for index, _ in enumerate(self.prop_locations):
            self.prop_locations[index] = self.propeller[index].prop_location
            self.prop_radii[index] = self.propeller[index].prop_radius

        self.vlm_mesh = meshing(span=self.wing.span,
                                chord=self.wing.chord[0],
                                prop_locations=self.prop_locations,
                                prop_radii=self.prop_radii,
                                nr_props=self.nr_props,
                                spanwise_discretisation_wing=self.spanwise_discretisation_wing,
                                spanwise_discretisation_propeller=self.spanwise_discretisation_propeller,
                                total_nodes=self.spanwise_discretisation_nodes)

        self.vlm_mesh_control_points = np.zeros(
            self.spanwise_discretisation_nodes-1, order='F')

        for panel in range(self.spanwise_discretisation_nodes-1):
            self.vlm_mesh_control_points[panel] = 0.5 * \
                (self.vlm_mesh[0, panel, 1]+self.vlm_mesh[0, panel+1, 1])