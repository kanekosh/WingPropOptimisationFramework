# --- Built-ins ---
from collections import OrderedDict

# --- Internal ---

# --- External ---
import numpy as np


def prop_circle(r, x)->list:
    y = np.sqrt(r**2-x**2)
    return y[np.where(y>=0)]

def get_niceColors():
    # Define an ordered dictionary of some nice Doumont style colors to use as the default color cycle
    niceColors = OrderedDict()
    niceColors["Yellow"] = "#e29400ff"  # '#f8a30dff'
    niceColors["Blue"] = "#1E90FF"
    niceColors["Red"] = "#E21A1A"
    niceColors["Green"] = "#00a650ff"
    niceColors["Maroon"] = "#800000ff"
    niceColors["Orange"] = "#ff8f00"
    niceColors["Purple"] = "#800080ff"
    niceColors["Cyan"] = "#00A6D6"
    niceColors["Black"] = "#000000ff"
    # The 2 colours below are not used in the colour cycle as they are too close to the other colours.
    # Grey is kept in the dictionary as it is used as the default axis/tick colour.
    # RedOrange is the old Orange, and is kept because I think it looks nice.
    niceColors["Grey"] = "#5a5758ff"
    niceColors["RedOrange"] = "#E21A1A"

    return niceColors


def get_delftColors():
    # Define an ordered dictionary of the official TU Delft colors to use as the default color cycle
    delftColors = OrderedDict()
    delftColors["Cyan"] = "#00A6D6"  # '#f8a30dff'
    delftColors["Yellow"] = "#E1C400"
    delftColors["Purple"] = "#6D177F"
    delftColors["Red"] = "#E21A1A"
    delftColors["Green"] = "#A5CA1A"
    delftColors["Blue"] = "#1D1C73"
    delftColors["Orange"] = "#E64616"
    delftColors["Grey"] = "#5a5758ff"
    delftColors["Black"] = "#000000ff"

    return delftColors

def get_SuperNiceColors():
    superniceColors = OrderedDict()
    superniceColors["Green"] = "#50D0E0"
    superniceColors["Purple"] = "#E06050"

    return superniceColors