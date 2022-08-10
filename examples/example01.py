"""example01.py module."""

import pandas as pd
from pynewmarkdisp.permanent_disp import classical_newmark, plot_permanent_disp

domain = "https://raw.githubusercontent.com/eamontoyaa/data4testing/main/pynewmarkdisp/"
file = "earthquake_data_simple.csv"
data = pd.read_csv(f"{domain}{file}", sep=";")

permanent_disp, data2plot = classical_newmark(
    data["Time"], data["Acceleration"], ky=0.2, g=1, step=1
)
fig = plot_permanent_disp(data2plot)
