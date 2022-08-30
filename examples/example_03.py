"""example_02.py module.

Example with dummy earthquake signal
"""

import time as time_pkg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as time_pkg
from pynewmarkdisp.newmark import classical_newmark, plot_newmark_str
from pynewmarkdisp.infslope import factor_of_safety, get_ky
from pynewmarkdisp.spatial import *

url = "https://raw.githubusercontent.com/eamontoyaa/data4testing/main/pynewmarkdisp/"
# Loading earthquake data
# earthquake = pd.read_csv(f"{url}earthquake_data_simple.csv", sep=";")
earthquake = pd.read_csv(f"{url}earthquake_data_real.csv", sep=";")
time = np.array(earthquake["Time"])
accel = np.array(earthquake["Acceleration"])
# g = 1.0  # It means, accel units are given in fractions of gravity
g = 981  # cm/s2 → This is for earthquake_data_real.csv data

# Loading spatial data
slope, header = load_ascii_raster(f"{url}example_02/slope.asc")
zones, header = load_ascii_raster(f"{url}example_02/zones.asc")
depth, header = load_ascii_raster(f"{url}example_02/zmax.asc")
depth[np.where(depth == 0)] = 0.1
depth_w, header = load_ascii_raster(f"{url}example_02/depthwt.asc")


# Performing the analysis
time_start = time_pkg.perf_counter()
# Defining zones' parameters
parameters = {  # Zone: frict_angle, cohesion, unit_weight
    1: (35, 3.5, 22),
    2: (31, 8, 22),
}
xy_lowerleft = (header["xllcorner"], header["yllcorner"])
# Creating arrays of material parameters
phi, c, unit_weight = map_zones(parameters, zones)
fig = plot_spatial_field(zones, xy_lowerleft, 10, "Zones", "Dark2")
# Calculating the factor of safety
fs = factor_of_safety(depth, depth_w, slope, phi, c, unit_weight, k_s=0.0)
fig = plot_spatial_field(fs, xy_lowerleft, 10, "$f_\mathrm{s}$", "RdYlGn")
# Critical seismic coeficient
ky = get_ky(depth, depth_w, slope, phi, c, unit_weight)
fig = plot_spatial_field(ky, xy_lowerleft, 10, "$k_\mathrm{y}$", "RdYlGn_r")
# Calculating permanent displacements
permanent_disp = spatial_newmark(time, accel, ky, g, step=4)
fig = plot_spatial_field(
    permanent_disp, xy_lowerleft, 10, "$\delta_\mathrm{perm}$  [m]", "RdYlGn_r"
)
time_end = time_pkg.perf_counter()
print(f"Elapsed time: {np.around(time_end - time_start, 4)} s")

# Verification
cell = (2, 2)
newmark_str = verify_spatial(
    cell, time, accel, g, depth, depth_w, slope, phi, c, unit_weight
)
fig = plot_newmark_str(newmark_str)
plt.show()
