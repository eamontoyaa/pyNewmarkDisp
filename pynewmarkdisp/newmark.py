# pynewmarkdisp/newmark.py

"""Functions to calculate permanent displacements using empirical correlations.

Permanent displacements are calculated based on the
[Newmark (1965)](https://doi.org/10.1680/geot.1965.15.2.139) sliding block
method.

"""

import numpy as np
from scipy.integrate import cumulative_trapezoid as cumul_integral
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numba import njit

from pynewmarkdisp.empir_corr import correlations

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


@njit(fastmath=True, cache=True)
def trapz(y, x):
    """Calculate the area under de curve y=f(x) by the trapezoidal rule method.

    This funtion is optimized for speed using Numba.

    Parameters
    ----------
    y : (n, ) ndarray
        1D array with the values of the function y=f(x).
    x : (n, ) ndarray
        1D array with the values of the independent variable x.

    Returns
    -------
    area : float
        Area under the curve y=f(x) calculated by the trapezoidal rule method.
    """
    return 0.5 * ((x[1:] - x[:-1]) * (y[1:] + y[:-1])).sum()


@njit(fastmath=True, cache=True)
def first_newmark_integration(time, accel, ay):
    """Perform the first integration of the Newmark method.

    The integration is performed using the trapezoidal rule. It accounts for
    energy dissipation at each peak above the critical acceleration by
    subtracting the area between $y=a_y$ and $y=a(t)$ after the peak until the
    velocity is zero.

    This funtion is optimized for speed using Numba.

    Parameters
    ----------
    time : (n, ) ndarray
        1D array with the time series of the earthquake record, in units
        consistent with ``accel`` and ``ay``.
    accel : (n, ) ndarray
        1D array with the acceleration series of the earthquake record, in the
        same units as ``ay``.
    ay : int or float
        Critical acceleration, in the same units as ``accel``.

    Returns
    -------
    vel : (n, ) ndarray
        1D array with the velocity series obtained by the first integration of
        the Newmark method. Units are consistent with ``accel`` and ``ay``.
    """
    length = len(time)
    vel = np.zeros(length)
    for i in np.arange(1, length, 1):
        if accel[i] > ay:
            v = vel[i - 1] + trapz(
                y=accel[i - 1 : i + 1] - ay, x=time[i - 1 : i + 1]
            )
        elif accel[i] < ay and vel[i - 1] > 0:
            v = vel[i - 1] - abs(
                trapz(y=accel[i - 1 : i + 1], x=time[i - 1 : i + 1])
            )
        else:
            v = 0
        vel[i] = max(v, 0)
    return vel


def direct_newmark(time, accel, ky, g):
    """
    Calculate the permanent displacements using the direct Newmark method.

    Parameters
    ----------
    time : (n,) ndarray
        1D array with the time series of the earthquake record, in [s].
    accel : (n,) ndarray
        1D array with the acceleration series of the earthquake record.
    ky : float
        Critical seismic coefficient (yield coefficient).
    g : float
        Acceleration of gravity, in the same units as ``accel``. For example,
        if ``accel`` is in [m/s²], ``g`` should be equal to 9.81; if ``accel``
        is as a fraction of gravity, ``g`` should be equal to 1.

    Returns
    -------
    newmark_str : dict
        Dictionary with the structure from the Newmark's method. The structure
        includes time, acceleration, velocity, and displacements series, as
        well as the critical seismic coefficient and accelerarion, and the
        permanent displacement calculated (in meters).
    """
    length = len(time)
    ay = 9.81 * ky  # Yield acceleration to SI units
    accel = 9.81 * accel / g  # Earthquake acceleration to SI units
    if ay == 0 or np.isnan(
        ay
    ):  # The slope is already unstable under static condition
        vel = np.full(length, np.nan)
        disp = np.full(length, np.nan)
    elif accel.max() > ay:  # Evaluate only if the earthquake exceeds `ay`
        # Velocity
        vel = first_newmark_integration(time, accel, ay)
        # Displacement
        disp = cumul_integral(y=vel, x=time, initial=0)
    else:  # The slope is stable even under seudostatic condition
        vel = np.zeros_like(accel)
        disp = np.zeros_like(accel)

    return {
        "time": time,
        "accel": accel,
        "vel": vel,
        "disp": disp,
        "ky": ky,
        "ay": ay,
        "perm_disp": disp[-1],
    }


def empirical_correlations(**kwargs):
    """Run a function of an empirical correlation for uₚ calculation.

    It is possible to call a function from the ``empir_corr.py`` module by
    passing the ID of the function to the ``opt`` keyword argument, or by
    passing the function itself to the ``correlation_funct`` keyword argument.

    The ``opt`` keyword argument follows the numbering of the functions in the
    Table 1 of [Meehan & Vahedifard (2013)](https://doi.org/10.1016/j.enggeo.2012.10.016)
    (see the ``empir_corr.py`` module for more details).

    After choosing either ``opt`` or ``correlation_funct``, the remaining
    keyword arguments are passed to the chosen function.

    Returns
    -------
    perm_disp : float
        Permanent displacement calculated by the chosen function using an
        empirical correlation.
    """
    if "opt" in kwargs:
        correlation_funct = correlations[kwargs["opt"]]
    else:
        correlation_funct = kwargs["correlation_funct"]
    return correlation_funct(**kwargs)


def plot_newmark_integration(newmark_str, compressed=False):
    """
    Plot the double-integration procedure of the direct Newmark method.

    Parameters
    ----------
    newmark_str : dict
        Dictionary with the structure from the Newmark's method. The structure
        includes time, acceleration, velocity, and displacements series, as
        well as the critical seismic coefficient and accelerarion, and the
        permanent displacement calculated (in meters). This dictionary is
        returned by the ``direct_newmark`` function.
    compressed : bool, optional
        If ``True``, the plot will be compressed in a single row. Otherwise,
        the plot will be in three rows. Default is ``False``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib object which might be used to save the figure as a file.
    """
    if compressed:
        fig, ax0 = plt.subplots(ncols=1, nrows=1, figsize=[6.5, 3])
        ax1 = ax0.twinx()
        ax2 = ax0.twinx()
        colors = ["#997700", "#BB5566", "#004488"]
    else:
        fig, axs = plt.subplots(
            ncols=1, nrows=3, figsize=[6.5, 5], sharex=True
        )
        ax0, ax1, ax2 = axs
        colors = ["#004488", "#004488", "#004488"]
    # Acceleration
    ax0.plot(newmark_str["time"], newmark_str["accel"], lw=0.75, c=colors[0])
    ay = newmark_str["ay"]
    ky = ay / 9.81
    l1 = ax0.axhline(
        ay,
        c="k",
        ls="--",
        label="$a_\\mathrm{y}=$" + f"{ky:.2f}g",  # = " + f"{ay:.2f} m/s$^2$",
    )
    ax0.set(ylabel="Acceleration, $a$ [m s$^{- 2}$]")
    # Velocity
    ax1.plot(newmark_str["time"], newmark_str["vel"], lw=0.75, color=colors[1])
    ax1.set(ylabel="Velocity, $v$ [m s$^{- 1}$]")
    # Displacement
    lw = 1.2 if compressed else 1.0
    ax2.plot(
        newmark_str["time"],
        newmark_str["disp"],
        lw=lw,
        c=colors[2],
    )
    (p1,) = ax2.plot(
        newmark_str["time"][-1],
        newmark_str["disp"][-1],
        ls="",
        marker="*",
        mfc="white",
        c=colors[2],
        label="$u_\\mathrm{p}=$" + f"{newmark_str['disp'][-1]:.3f} m",
    )
    ax2.set(ylabel="Displacement, $u$ [m]", xlabel="Time, $t$ [s]")
    # Setup
    for ax, color in zip([ax0, ax1, ax2], colors):
        color_lbl = color if compressed else "k"
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.grid(False)
        ax.yaxis.label.set_color(color_lbl)
        ax.tick_params(axis="y", colors=color_lbl)
    if compressed:  # Adjust the position of the axis and its label
        ax1.spines["left"].set_visible(True)  # Make the spine visible
        ax1.spines["left"].set_position(("outward", 50))
        ax1.yaxis.set_label_position("left")
        ax1.yaxis.set_ticks_position("left")  # Position the ticks on the left
        ax0.grid(True, which="major", linestyle="--")
        fig.legend(
            handles=[l1, p1],
            loc="lower right",
            bbox_to_anchor=(0.9, 0.2),
        )
        ax0.set(xlabel="Time, $t$ [s]")
    else:
        [ax.grid(True, which="major", linestyle="--") for ax in axs]
        ax0.legend(loc="best")
        ax2.legend(loc="lower right")
    fig.tight_layout(pad=0.1)
    return fig
