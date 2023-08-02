# --- Built-ins ---
from dataclasses import dataclass


@dataclass
class WingInfo:
    span: float


@dataclass
class PropInfo:
    nr_blades: int
    

@dataclass    
class WingPropInfo(PropInfo, WingInfo):
    nr_props: int
