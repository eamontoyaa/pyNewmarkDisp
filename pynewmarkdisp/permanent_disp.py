"""Compute permanent displacements by the Newmark method"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("default")
mpl.rcParams.update(
    {
        "text.usetex": False,  # Use mathtext, not LaTeX
        "font.family": "serif",  # Use the Computer modern font
        "font.serif": "cmr10",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        # "backend": "TKAgg",
    }
)


class Data2Plot:
    pass


def classical_newmark(t, accel, ky, g, step=1):
    ay = 9.81 * ky  # Yield acceleration to SI units
    accel = 9.81 * np.array(accel) / g  # Earthquake acceleration to SI units
    if step > 1:
        indices = np.arange(0, len(t), step)
        t = t[indices]
        accel = accel[indices]
    vel = [0]
    disp = [0]

    for i in np.arange(1, len(t), 1):
        if accel[i] > ay:
            v = vel[-1] + np.trapz(
                y=accel[i - 1 : i + 1] - ay, x=t[i - 1 : i + 1]
            )
        elif accel[i] < ay and vel[-1] > 0:
            v = vel[-1] - abs(
                np.trapz(y=accel[i - 1 : i + 1], x=t[i - 1 : i + 1])
            )
        else:
            v = 0
        if v < 0:
            v = 0
        vel.append(v)
        disp.append(np.trapz(y=vel, x=t[: i + 1]))
        data2plot = Data2Plot()
        data2plot.t = t
        data2plot.accel = accel
        data2plot.vel = vel
        data2plot.disp = disp
        data2plot.ay = ay
        permanent_disp = disp[-1]
    return permanent_disp, data2plot


def plot_permanent_disp(data2plot):
    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=[6, 6], sharex=True)
    axs[0].plot(data2plot.t, data2plot.accel, lw=1.5, color="#EC3D33")
    axs[0].axhline(
        data2plot.ay,
        color="k",
        ls="--",
        label="$a_\mathrm{y}=$" + f"{data2plot.ay/9.81}g",
    )
    axs[0].set(ylabel="Acceleration, $a$ [m s$^{- 2}$]")
    axs[0].legend(loc="best")

    axs[1].plot(data2plot.t, data2plot.vel, lw=1.5, color="#F5B12E")
    axs[1].set(ylabel="Velocity, $v$ [m s$^{- 1}$]")

    axs[2].plot(
        data2plot.t,
        data2plot.disp,
        lw=1.5,
        color="#234990",
        label="$\delta_\mathrm{perm}=$" + f"{data2plot.disp[-1]:.3f} m",
    )
    axs[2].set(ylabel="Displacement, $\delta$ [m]", xlabel="Time, $t$ [s]")
    axs[2].legend(loc="best")

    for ax in axs.flat:
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.grid(True, which="major", linestyle="--")
    fig.tight_layout()
    return fig
