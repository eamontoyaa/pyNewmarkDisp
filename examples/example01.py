"""example01.py module."""

import pandas as pd
import time

from pynewmarkdisp.permanent_disp import classical_newmark, plot_permanent_disp

domain = "https://raw.githubusercontent.com/eamontoyaa/data4testing/main/pynewmarkdisp/"
# file = "earthquake_data_simple.csv"
file = "earthquake_data_real.csv"
data = pd.read_csv(f"{domain}{file}", sep=";")


time_start = time.time()
accel, vel, disp = classical_newmark(
    data["Time"], data["Acceleration"], ky=0.2, g=981, step=1
)
time_end = time.time()
print(time_end - time_start)


class Data2Plot:
    pass


data2plot = Data2Plot()
data2plot.t = data["Time"]
data2plot.accel = accel
data2plot.vel = vel
data2plot.disp = disp
data2plot.ay = 9.81 * 0.2
permanent_disp = disp[-1]

fig = plot_permanent_disp(data2plot)
fig.show()
fig.savefig("./examples/example01.pdf")
