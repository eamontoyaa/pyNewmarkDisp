"""example_02.py module.

Example with dummy earthquake signal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as time_pkg
from pynewmarkdisp.newmark import classical_newmark, plot_newmark_str
from pynewmarkdisp.infslope import factor_of_safety, get_ky
from pynewmarkdisp.spatial import map_zones

url = "https://raw.githubusercontent.com/eamontoyaa/data4testing/main/pynewmarkdisp/"
# Loading earthquake data
earthquake = pd.read_csv(f"{url}earthquake_data_simple.csv", sep=";")
# Earthquake
time = np.array(earthquake["Time"])
accel = np.array(earthquake["Acceleration"])
g = 1.0  # It means, accel units are given in fractions of gravity

depth = 3  # m; depth of planar landslide
depth_w = 2.5  # m; depth of watertable
beta = 25  # °; slope angle
unit_weight = 17  # kN/m3; unit weigth of the material
phi = 27  # °; friction angle
c = 10  # kPa; cohesion
k_s = 0.0  # -; seismic coefficient

fs = factor_of_safety(depth, depth_w, beta, phi, c, unit_weight, k_s=0.0)
print(f"Static factor of safety: {fs}")
ky = get_ky(depth, depth_w, beta, phi, c, unit_weight)
print(f"Critical seismic coefficient: {ky}")
fs = factor_of_safety(depth, depth_w, beta, phi, c, unit_weight, k_s=ky)
print(f"Pseudostatic factor of safety: {fs}")
newmark_str = classical_newmark(time, accel, ky, g, step=1)
print(f"Permanent displacement: {newmark_str['perm_disp']:.3f} m")
fig = plot_newmark_str(newmark_str)
plt.show()
