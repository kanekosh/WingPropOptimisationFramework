# --- Built-ins ---
from dataclasses import dataclass

# --- Internal ---
from src.utils.meshing import meshing

# --- External ---
import numpy as np


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
    hub_offset: float = 0.05
        
    def __post_init__(self):
        self.prop_radius = np.zeros(len(self.span+1))
        for index, ispan in enumerate(self.span):
            # TODO: this only works for a linearly spaced propeller
            self.prop_radius[index+1] = index*ispan

    def __str__(self):
        return f'Propeller {self.label}, with {self.nr_blades} blades'


@dataclass
class WingPropInfo:
    nr_props: int
    spanwise_discretisation_wing: int
    spanwise_discretisation_propeller: int
    
    propeller: list[PropInfo]
    wing: WingInfo

    def __post_init__(self):
        self.spanwise_discretisation = self.spanwise_discretisation_wing + \
            self.nr_props*self.spanwise_discretisation_propeller + 1
        
        self.prop_locations = np.zeros((self.nr_props), order='F')
        self.prop_radii = np.zeros((self.nr_props, self.spanwise_discretisation_propeller), order='F')
        
        for index, _ in enumerate(self.prop_locations):
            self.prop_locations[index] = self.propeller[index].prop_location
            self.prop_radii[index] = self.propeller[index].prop_radius