import numpy as np


def factor_of_safety(depth, depth_w, beta, phi, c, unit_weight, k_s=0):
    """Calculate the factor of safety for an infinite slope mechanism.

    Inputs can be either constants or spatially variable.

    Parameters
    ----------
    depth : float or ndarray
        Depth of the planar suface [m]
    depth_w : float or ndarray
        Depth of the watertable [m]
    beta : float or ndarray
        Slope inclination [°]
    phi : float or ndarray
        Slope inclination [°]
    c : float or ndarray
        Cohesion [kPa]
    unit_weight : float or ndarray
        Mean unit weigth of the soil [kN/m3]
    k_s : float or ndarray
        Seismic coefficient for pseudostatic analysis

    Returns
    -------
    float or ndarray
        Factor of safety
    """
    beta = np.radians(beta)  # Slope angle to radians
    phi = np.radians(phi)  # Friction angle to radians
    if isinstance(depth_w, np.ndarray):  # Watertable not lower than slip surf.
        depth_w[np.where(depth_w > depth)] = depth[np.where(depth_w > depth)]
    elif isinstance(depth_w, (int, float)) and depth_w > depth:
        depth_w = depth
    head_pressure = (depth - depth_w) * np.cos(beta) ** 2
    pw = head_pressure * 9.81  # kN; Hydrostatic force
    w = depth * unit_weight * np.cos(beta)  # kPa; Weight of sliding mass
    tau = (w * (np.cos(beta) - k_s * np.sin(beta)) - pw) * np.tan(phi) + c
    slip_force = w * (np.sin(beta) + k_s * np.cos(beta))
    if isinstance(slip_force, np.ndarray):  # Slip force cannot be zero
        slip_force[np.where(slip_force == 0)] = 0.01
    elif isinstance(slip_force, (int, float)) and slip_force <= 0:
        slip_force = 0.01
    fs = tau / slip_force
    if isinstance(fs, np.ndarray):  # Factor of safety restricted to 5 max.
        fs[np.where(fs > 3)] = 3
    elif isinstance(fs, (int, float)) and fs > 3:
        fs = 3
    return np.around(fs, 3)


def get_ky(depth, depth_w, beta, phi, c, unit_weight):
    """Calculate the critical seismic coefficient of an infinite slope model.

    Inputs can be either constants or spatially variable.

    Parameters
    ----------
    depth : float or ndarray
        Depth of the planar suface [m]
    depth_w : float or ndarray
        Depth of the watertable [m]
    beta : float or ndarray
        Slope inclination [°]
    unit_weight : float or ndarray
        Mean unit weigth of the soil [kN/m3]
    phi : float or ndarray
        Slope inclination [°]
    c : float or ndarray
        Cohesion [kPa]

    Returns
    -------
    float or ndarray
        Factor of safety
    """
    beta = np.radians(beta)  # Slope angle to radians
    phi = np.radians(phi)  # Friction angle to radians
    if type(depth_w) == np.ndarray:  # Watertable at slip surface if below it
        depth_w[np.where(depth_w > depth)] = depth[np.where(depth_w > depth)]
    elif any((type(depth_w) == int, float)) and depth_w > depth:
        depth_w = depth
    head_pressure = (depth - depth_w) * (np.cos(beta)) ** 2
    pw = head_pressure * 9.81  # kN; Hydrostatic force
    w = depth * unit_weight * np.cos(beta)  # kPa; Weight of sliding mass
    # CALCULATION OF Ky
    num = (np.cos(beta) - pw / w) * np.tan(phi) + c / w - np.sin(beta)
    den = np.cos(beta) + np.sin(beta) * np.tan(phi)
    ky = num / den
    if isinstance(
        ky, np.ndarray
    ):  # The slope is unstable under static condition
        ky[np.where(ky < 0)] = 0
    elif isinstance(ky, (int, float)) and ky < 0:
        ky = 0
    return np.around(ky, 3)
