# pynewmarkdisp_/infslope.py

"""Functions to calculate the factor of safety of an infinite slope mechanism.

"""

import numpy as np


def num_to_2darray(num):
    """Turn an int or float input into a 2D array with shape (1, 1).

    Parameters
    ----------
    num : int or float
        Variable to be converted to a 2D array

    Returns
    -------
    num_array : ndarray
        2D array with shape (1, 1) whose only element is the input variable.
    """
    return (
        np.array(num).reshape((1, 1)) if isinstance(num, (int, float)) else num
    )


def factor_of_safety(depth, depth_w, slope, phi, c, gamma, ks=0):
    """Calculate the factor of safety for an infinite slope mechanism.

    Inputs can be either constant or spatially distributed variables.

    Parameters
    ----------
    depth : float or (m, n) ndarray
        Depth of the planar suface, in [m].
    depth_w : float or ndarray
        Depth of the watertable, in [m].
    slope : float or ndarray
        Slope inclination, in [°].
    phi : float or ndarray
        Slope inclination, in [°].
    c : float or ndarray
        Cohesion, in [kPa].
    gamma : float or ndarray
        Average unit weigth of the soil, in [kN/m³].
    ks : float or ndarray
        Seismic coefficient for pseudostatic analysis. Its default value is
        ``0``, meaning that the analysis is static (no seismic force included).

    Returns
    -------
    fs : float or ndarray
        Factor of safety
    """
    depth = num_to_2darray(depth)
    depth_w = num_to_2darray(depth_w)
    slope = num_to_2darray(np.radians(slope))  # Slope angle to radians
    phi = num_to_2darray(np.radians(phi))  # Friction angle to radians
    # Calculation
    # depth[np.where(depth == 0)] = np.nan  # Slice height cannot be zero
    head_pressure = (depth - depth_w) * np.cos(slope) ** 2
    head_pressure[np.where(depth_w > depth)] = 0  # Watertable below slip surf.
    pw = head_pressure * 9.81  # kN; Hydrostatic force
    w = depth * gamma * np.cos(slope)  # kPa; Weight of sliding mass
    tau = (w * (np.cos(slope) - ks * np.sin(slope)) - pw) * np.tan(phi) + c
    slip_force = w * (np.sin(slope) + ks * np.cos(slope))
    # slip_force[np.where(beta == 0)] = 0.001  # ~zero for horizontal surfaces
    np.seterr(divide="ignore")
    fs = np.divide(tau, slip_force)
    fs[np.where(fs > 3)] = 3  # Factor of safety restricted to max. 3
    if fs.size == 1:
        fs = fs[0, 0]
    return np.around(fs, 3)


def min_fs_and_depth(depth, depth_w, slope, phi, c, gamma, ks=0, nd=10):
    """Calculate the minimum factor of safety and the depth at which it occurs.

    As for the ``factor_of_safety`` function, inputs can be either constant or
    spatially distributed variables.

    It evaluates the factor of safety for a range of ``nd`` depths between the
    terrain surface and the slip surface and return the minimum factor of
    safety and the depth at which it occurs. This is useful to find the
    critical slip surface, which will be located at the first depth where the
    factor of safety drops below 1 or where it reaches its minimum value.

    Parameters
    ----------
    depth : float or (m, n) ndarray
        Depth of the planar suface, in [m].
    depth_w : float or ndarray
        Depth of the watertable, in [m].
    slope : float or ndarray
        Slope inclination, in [°].
    phi : float or ndarray
        Slope inclination, in [°].
    c : float or ndarray
        Cohesion, in [kPa].
    gamma : float or ndarray
        Average unit weigth of the soil, in [kN/m³].
    ks : float or ndarray
        Seismic coefficient for pseudostatic analysis. Its default value is
        ``0``, meaning that the analysis is static (no seismic force included).
    nd : int, optional
        Number of depths to evaluate between the terrain surface and the slip
        surface. The default is ``10``.

    Returns
    -------
    min_fs : float or ndarray
        Minimum factor of safety
    depth_at_min_fs : float or ndarray
        Depth at which the minimum factor of safety occurs, in [m].
    """
    depth = num_to_2darray(depth)
    # depth[np.where(depth == 0)] = np.nan  # Slice height cannot be zero
    min_fs = factor_of_safety(depth, depth_w, slope, phi, c, gamma, ks)
    depth_at_min_fs = depth.copy()
    for ith_depth_fract in np.linspace(0, 1, nd + 1)[1:]:
        ith_depth = ith_depth_fract * depth
        ith_fs = factor_of_safety(ith_depth, depth_w, slope, phi, c, gamma, ks)
        mask = np.where(ith_fs < min_fs)
        min_fs[mask] = ith_fs[mask]
        depth_at_min_fs[mask] = ith_depth[mask]
    return min_fs, depth_at_min_fs


def get_ky(depth, depth_w, slope, phi, c, gamma):
    """Compute the critical seismic coefficient of an infinite slope mechanism.

    Inputs can be either constant or spatially distributed variables.

    Parameters
    ----------
    depth : float or (m, n) ndarray
        Depth of the planar suface, in [m].
    depth_w : float or ndarray
        Depth of the watertable, in [m].
    slope : float or ndarray
        Slope inclination, in [°].
    phi : float or ndarray
        Slope inclination, in [°].
    c : float or ndarray
        Cohesion, in [kPa].
    gamma : float or ndarray
        Average unit weigth of the soil, in [kN/m³].

    Returns
    -------
    ky : float or ndarray
        Critical seismic coefficient.
    """
    depth = num_to_2darray(depth)
    depth_w = num_to_2darray(depth_w)
    slope = np.radians(slope)  # Slope angle to radians
    phi = np.radians(phi)  # Friction angle to radians
    # Intermediate calculations
    # depth[np.where(depth == 0)] = np.nan  # Slice height cannot be zero
    head_pressure = (depth - depth_w) * np.cos(slope) ** 2
    pw = head_pressure * 9.81  # kN; Hydrostatic force
    w = depth * gamma * np.cos(slope)  # kPa; Weight of sliding mass
    # Ky calculation
    np.seterr(divide="ignore")
    # num = np.divide(np.cos(beta) - pw, w) * np.divide(
    #     np.tan(phi) + c, w
    # ) - np.sin(beta)
    w[np.where(w == 0)] = 0.001
    num = (np.cos(slope) - pw / w) * np.tan(phi) + c / w - np.sin(slope)
    den = np.cos(slope) + np.sin(slope) * np.tan(phi)
    ky = np.divide(num, den)
    # ky = num / den
    ky[np.where(ky > 3)] = 3  # The slope is stable under pseudo-static cond.
    ky[np.where(ky < 0)] = np.nan  # The slope is unstable under static cond.
    # ky[np.where(ky > 1)] = 1  # The slope is stable under sedudostatic cond.
    if ky.size == 1:
        ky = ky[0, 0]
    return np.around(ky, 3)
