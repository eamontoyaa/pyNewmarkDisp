# -*- coding: utf-8 -*-
"""
Functions for calculating Newmark's displacements
-------------------------------------------------

Functions to compute the permanent displacements by the Newmark method.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid as cum_trapz
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numba import njit

plt.style.use("default")
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

# @njit(cache=True)
def classical_newmark(time, accel, ky, g, step=1):
    """
    Calculate Newmark's displacements by the classsical approach.

    Parameters
    ----------
    time : ndarray
        Array with the time steps of the accelerogram.
    accel : ndarray
        Array with the acceleration of each time step of the accelerogram.
    ky : float
        Critical seismic coefficient (yield coefficient).
    g : float
        Gravitational acceleration in the same units as `accel`.  If `accel` is
        given in fractions  of g, g=1.
    step : int (optional)
        Value to not read all the data elements but each `step` indices.

    Returns
    -------
    vel : ndarray
        Array with the velocities after integrating the accelerations.
    disp : ndarray
        Array with the cumulative permanent displazaments after doublle
        integrating the accelerations.
    time : ndarray
        Resized array with the time steps of the accelerogram.  This output is
        returned only if `step != 1`.
    accel : ndarray
        Resized array with the time steps of the accelerogram.  This output is
        returned only if `step != 1`.
    """
    ay = 9.81 * ky  # Yield acceleration to SI units
    accel = 9.81 * accel / g  # Earthquake acceleration to SI units
    if step > 1:
        indices = np.arange(0, len(time), step)
        time = time[indices]
        accel = accel[indices]

    vel = np.zeros(len(time))
    for i in np.arange(1, len(time), 1):
        if accel[i] > ay:
            v = vel[i - 1] + np.trapz(
                y=accel[i - 1 : i + 1] - ay, x=time[i - 1 : i + 1]
            )
        elif accel[i] < ay and vel[i - 1] > 0:
            v = vel[i - 1] - abs(
                np.trapz(y=accel[i - 1 : i + 1], x=time[i - 1 : i + 1])
            )
        else:
            v = 0
        if v < 0:
            v = 0
        vel[i] = v
    disp = cum_trapz(y=vel, x=time, initial=0)
    newmark_str = {
        "time": time,
        "accel": accel,
        "vel": vel,
        "disp": disp,
        "ay": ay,
    }
    return newmark_str


def plot_newmark_str(newmark_str):
    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=[6, 5], sharex=True)
    axs[0].plot(
        newmark_str["time"], newmark_str["accel"], lw=0.5, color="#EC3D33"
    )
    ay = newmark_str["ay"]
    ky = ay / 9.81
    axs[0].axhline(
        newmark_str["ay"],
        color="k",
        ls="--",
        label="$a_\mathrm{y}=$" + f"{ky:.1f}g = " + f"{ay:.2f} m/s$^2$",
    )
    axs[0].set(ylabel="Acceleration, $a$ [m s$^{- 2}$]")
    axs[0].legend(loc="best")

    axs[1].plot(
        newmark_str["time"], newmark_str["vel"], lw=0.5, color="#EC3D33"
    )
    axs[1].set(ylabel="Velocity, $v$ [m s$^{- 1}$]")

    axs[2].plot(
        newmark_str["time"],
        newmark_str["disp"],
        lw=1.0,
        color="#EC3D33",
    )
    axs[2].plot(
        newmark_str["time"][-1],
        newmark_str["disp"][-1],
        ls="",
        marker="*",
        mfc="white",
        color="#EC3D33",
        label="$\delta_\mathrm{perm}=$" + f"{newmark_str['disp'][-1]:.3f} m",
    )
    axs[2].set(ylabel="Displacement, $\delta$ [m]", xlabel="Time, $t$ [s]")
    axs[2].legend(loc="lower right")

    for ax in axs.flat:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.grid(True, which="major", linestyle="--")
    fig.tight_layout()
    return fig
