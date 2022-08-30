# -*- coding: utf-8 -*-
"""
Functions for calculating Newmark's displacements
-------------------------------------------------

Functions to compute the permanent displacements by the Newmark method.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid as cum_trapz
from scipy.integrate import simpson
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from numba import njit, jit

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


@njit
def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


@njit
def cumulative_trapezoid(y, x, initial=None):
    axis = -1
    d = np.diff(x, axis=axis)
    nd = len(y.shape)
    slice1 = tupleset((slice(None),) * nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),) * nd, axis, slice(None, -1))
    res = np.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)
    shape = list(res.shape)
    shape[axis] = 1
    res = np.concatenate(
        [np.full(shape, initial, dtype=res.dtype), res], axis=axis
    )
    return res


# @jit(forceobj=True)
@njit
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
    length = len(time)
    if accel.max() > ay:  # Evaluate only if the earthquake exceeds `ay`
        if step > 1:
            indices = np.arange(0, length, step)
            time = time[indices]
            accel = accel[indices]
            length = len(time)

        vel = np.empty(length)
        for i in np.arange(1, length, 1):
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
            v = max(v, 0)
            vel[i] = v
        # disp = cum_trapz(y=vel, x=time, initial=0)
        disp = cumulative_trapezoid(y=vel, x=time, initial=0)
    else:
        vel = np.zeros(length)
        disp = np.zeros(length)

    return {
        "time": time,
        "accel": accel,
        "vel": vel,
        "disp": disp,
        "ky": ky,
        "ay": ay,
        "perm_disp": disp[-1],
    }


def plot_newmark_str(newmark_str):
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
