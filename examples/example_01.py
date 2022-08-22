"""example_01.py module."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as time_pkg

from pynewmarkdisp.newmark import classical_newmark, plot_newmark_str

# Example with dummy earthquake signal
earthquake = "https://raw.githubusercontent.com/eamontoyaa/data4testing/main/pynewmarkdisp/earthquake_data_simple.csv"
earthquake = pd.read_csv(earthquake, sep=";")
time = np.array(earthquake["Time"])
accel = np.array(earthquake["Acceleration"])
g = 1.0
ky = 0.2

time_start = time_pkg.perf_counter()
newmark_str = classical_newmark(time, accel, ky, g, step=1)
time_end = time_pkg.perf_counter()
print(time_end - time_start)

fig = plot_newmark_str(newmark_str)
plt.show()
# fig.savefig("./examples/example_01.pdf")


# Example with a real earthquake signal
earthquake = "https://raw.githubusercontent.com/eamontoyaa/data4testing/main/pynewmarkdisp/earthquake_data_real.csv"
earthquake = pd.read_csv(earthquake, sep=";")
time = np.array(earthquake["Time"])
accel = np.array(earthquake["Acceleration"])
g = 981  # m/s^2
ky = 0.15

time_start = time_pkg.perf_counter()
newmark_str = classical_newmark(time, accel, ky, g, step=1)
time_end = time_pkg.perf_counter()
print(time_end - time_start)

fig = plot_newmark_str(newmark_str)
plt.show()
# fig.savefig("./examples/example_01.pdf")
