# -*- coding: utf-8 -*-
"""
Functions for calculating Newmark's displacements
-------------------------------------------------

Functions to compute the permanent displacements by the Newmark method.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numba import njit, jit

from pynewmarkdisp.newmark import classical_newmark
from pynewmarkdisp.infslope import factor_of_safety, get_ky

# plt.style.use("default")
mpl.rcParams.update(
    {
        "text.usetex": False,  # Use mathtext, not LaTeX
        "font.family": "serif",  # Use the Computer modern font
        "font.serif": "cmr10",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "backend": "TKAgg",
    }
)


def map_zones(parameters, zones):
    phi = np.empty_like(zones)
    c = np.empty_like(zones)
    unit_weight = np.empty_like(zones)
    for z in parameters.keys():  # Zone: frict_angle, cohesion, unit_weight
        phi[np.where(zones == z)] = parameters[z][0]
        c[np.where(zones == z)] = parameters[z][1]
        unit_weight[np.where(zones == z)] = parameters[z][2]
    return phi, c, unit_weight


def spatial_newmark(time, accel, ky, g, step=1):
    row, col = ky.shape
    permanent_disp = np.empty_like(ky)
    for i in range(row):
        for j in range(col):
            if isinstance(accel, np.ndarray) and accel.ndim == 2:
                newmark_str = classical_newmark(
                    time[i, j], accel[i, j], ky[i, j], g, step
                )
            elif isinstance(accel, np.ndarray) and accel.ndim == 1:
                newmark_str = classical_newmark(time, accel, ky[i, j], g, step)
            permanent_disp[i, j] = newmark_str["perm_disp"]
    return np.around(permanent_disp, 3)


def verify_spatial(
    cell, time, accel, g, depth, depth_w, slope, phi, c, unit_weight
):
    i, j = cell
    depth = depth[i, j]
    depth_w = depth_w[i, j]
    slope = slope[i, j]
    phi = phi[i, j]
    c = c[i, j]
    unit_weight = unit_weight[i, j]
    fs_0 = factor_of_safety(depth, depth_w, slope, phi, c, unit_weight, k_s=0)
    ky = get_ky(depth, depth_w, slope, phi, c, unit_weight)
    fs_ky = factor_of_safety(depth, depth_w, slope, phi, c, unit_weight, ky)
    newmark_str = classical_newmark(time, accel, ky, g, step=1)
    newmark_str["fs_init"] = fs_0
    newmark_str["fs_ky"] = fs_ky
    return newmark_str


def plot_spatial_field(field, xy_lowerleft, cellsize, title=None, cmap="jet"):
    """
    Plot the double-integration process from Newmark's method.

    Parameters
    ----------
    newmark_str : dict
        Dictionary with the structure from the Newmark's method. The structure
        includes time, acceleration, velocity, displacements, and critical
        acceleration.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib object which might be used to save the figure as a file.
    """
    x, y = field.shape
    extent = (
        xy_lowerleft[0],  # x min
        xy_lowerleft[0] + cellsize * x,  # x max
        xy_lowerleft[1],  # y min
        xy_lowerleft[1] + cellsize * y,  # y max
    )
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[6, 5], sharex=True)
    im = ax.imshow(field, cmap=cmap, interpolation="nearest", extent=extent)
    ax.tick_params(axis="y", labelrotation=90)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.grid(True, which="major", linestyle="--")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(title, rotation=90, size="large")
    fig.tight_layout()
    return fig


def load_ascii_raster(path):
    raster = np.loadtxt(path, skiprows=6)
    header = np.loadtxt(path, max_rows=6, dtype=object)
    header = {
        "ncols": int(header[0, 1]),
        "nrows": int(header[1, 1]),
        "xllcorner": float(header[2, 1]),
        "yllcorner": float(header[3, 1]),
        "cellsize": float(header[4, 1]),
        "nodata_value": int(header[5, 1]),
    }
    return (raster, header)
