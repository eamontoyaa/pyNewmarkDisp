"""example_02.py module."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as time_pkg

# import richdem as rd

from pynewmarkdisp.newmark import classical_newmark, plot_newmark_str

# Example with dummy earthquake signal

# Loading external data
url = "https://raw.githubusercontent.com/eamontoyaa/data4testing/main/pynewmarkdisp/"
earthquake = pd.read_csv(url + "earthquake_data_simple.csv", sep=";")
slope = np.loadtxt(url + "input_files_example_02/slope.txt")
# slope = rd.rdarray(slope, no_data=-9999)
zones = np.loadtxt(url + "input_files_example_02/zones.txt")
# zones = rd.rdarray(slope, no_data=-9999)
print(zones)

# inputs
time = np.array(earthquake["Time"])
accel = np.array(earthquake["Acceleration"])
g = 1.0
ky = 0.2
parameters = {  # Zone: frict_angle, cohesion, unit_weigth
    1: (35, 3.5, 22),
    2: (31, 8, 22),
}
