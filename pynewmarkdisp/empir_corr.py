# ``pynewmarkdisp/empr_corr.py``

"""Functions to calculate permanent displacements from empirical correlations.

Most of the functions in this module are taken from Table 1 in
[Meehan & Vahedifard (2013)](https://doi.org/10.1016/j.enggeo.2012.10.016),
who compiled a several empirical correlations or simplified methods for
calculating the permanent displacement of a slope. The keys of the
`correlations` dictionary are the same as the numbering of the cited table.
However, additional functions were added to the dictionary after consulting
the original references.

"""


import numpy as np


def hynesgriffin_and_Franklin_1984_ub(ay, a_max, **kwargs):
    """Hynes-Griffin and Franklin (1984) empirical corr. for uₚ (upper bound).

    Eq. 7 in Meehan & Vahedifard (2013). It is a functional form of the upper
    bound resulting from regression analysis of chart-based solution from the
    original paper.

    Parameters
    ----------
    ay : float
        Critical acceleration, in fraction of g.
    a_max : float
        Peak horizontal ground acceleration, in fraction of g.

    Returns
    -------
    perm_disp : float
        Permanent displacement, in [m].
    """
    # ay[np.where(ay > a_max)] = a_max  # There is no excess acceleration
    car = ay / a_max  # Critical acceleration ratio
    return 1e-2 * 10 ** (
        0.804
        - 1.847 * car
        - 0.285 * car**2
        + 0.193 * car**3
        + 0.078 * car**4
    )


def hynesgriffin_and_Franklin_1984_mean(ay, a_max, **kwargs):
    """Hynes-Griffin and Franklin (1984) empirical correlation for uₚ (mean).

    Eq. 8 in Meehan & Vahedifard (2013). It is a functional form of the mean
    resulting from regression analysis of chart-based solution from the
    original paper.

    Parameters
    ----------
    ay : float
        Critical acceleration, in fraction of g.
    a_max : float
        Peak horizontal ground acceleration, in fraction of g.

    Returns
    -------
    perm_disp : float
        Permanent displacement, in [m].
    """
    # ay[np.where(ay > a_max)] = a_max  # There is no excess acceleration
    car = ay / a_max  # Critical acceleration ratio
    return 1e-2 * 10 ** (
        -0.287
        - 2.854 * car
        - 1.733 * car**2
        - 0.702 * car**3
        - 0.116 * car**4
    )


def ambraseys_and_menu_88(ay, a_max, **kwargs):
    """Ambraseys and Menu (1988) empirical correlation for uₚ.

    Eq. 10 in Meehan & Vahedifard (2013).

    Parameters
    ----------
    ay : float
        Critical acceleration, in fraction of g.
    a_max : float
        Peak horizontal ground acceleration, in fraction of g.

    Returns
    -------
    perm_disp : float
        Permanent displacement, in [m].
    """
    # ay[np.where(ay > a_max)] = a_max  # There is no excess acceleration
    car = ay / a_max  # Critical acceleration ratio
    return 1e-2 * 10**0.90 * ((1 - car) ** 2.53 * car ** (-1.09))


def bray_and_travasarou_07_eq6(ay, a_max, magnitude, **kwargs):
    """Bray and Travasarou (2007) empirical correlation for uₚ.

    Eq. 10 in Meehan & Vahedifard (2013). Eq. 6 in Bray and Travasarou (2007).

    Parameters
    ----------
    ay : float
        Critical acceleration, in fraction of g.
    a_max : float
        Peak horizontal ground acceleration, in fraction of g.
    magnitude : float
        Earthquake magnitude.

    Returns
    -------
    perm_disp : float
        Permanent displacement, in [m].
    """
    # ay[np.where(ay > a_max)] = a_max  # There is no excess acceleration
    car = ay / a_max  # Critical acceleration ratio
    return 1e-2 * np.exp(
        -0.22
        - 2.83 * np.log(ay)
        - 0.333 * (np.log(ay)) ** 2
        + 0.566 * np.log(ay) * np.log(a_max)
        + 3.04 * np.log(a_max)
        - 0.244 * (np.log(a_max)) ** 2
        + 0.278 * (magnitude - 7)
    )


def jibson_07_eq6(ay, a_max, **kwargs):
    """Jibson (2007) empirical correlation for uₚ.

    Eq. 16 in Meehan & Vahedifard (2013). Eq. 6 in Jibson (2007).

    Parameters
    ----------
    ay : float
        Critical acceleration, in fraction of g.
    a_max : float
        Peak horizontal ground acceleration, in fraction of g.

    Returns
    -------
    perm_disp : float
        Permanent displacement, in [m].
    """
    # ay[np.where(ay > a_max)] = a_max  # There is no excess acceleration
    car = ay / a_max  # Critical acceleration ratio
    return 1e-2 * 10 ** (0.215) * ((1 - ay / a_max) ** 2.341 * car ** (-1.438))


def jibson_07_eq9(ay, arias_int, **kwargs):
    """Jibson (2007) empirical correlation for uₚ.

    Eq. 18 in Meehan & Vahedifard (2013). Eq. 9 in Jibson (2007).

    Parameters
    ----------
    ay : float
        Critical acceleration, in fraction of g.
    arias_int : float
        Arias intensity, in [m/s].

    Returns
    -------
    perm_disp : float
        Permanent displacement, in [m].
    """
    # ay[np.where(ay > a_max)] = a_max  # There is no excess acceleration
    return 1e-2 * 10 ** (
        2.401 * np.log10(arias_int) - 3.481 * np.log10(ay) - 3.23
    )


def jibson_07_eq10(ay, a_max, arias_int, **kwargs):
    """Jibson (2007) empirical correlation for uₚ.

    Eq. 19 in Meehan & Vahedifard (2013). Eq. 10 in Jibson (2007).

    Parameters
    ----------
    ay : float
        Critical acceleration, in fraction of g.
    a_max : float
        Peak horizontal ground acceleration, in fraction of g.
    arias_int : float
        Arias intensity, in [m/s].

    Returns
    -------
    perm_disp : float
        Permanent displacement, in [m].
    """
    # ay[np.where(ay > a_max)] = a_max  # There is no excess acceleration
    car = ay / a_max  # Critical acceleration ratio
    return 1e-2 * 10 ** (
        0.561 * np.log10(arias_int) - 3.833 * np.log10(car) - 1.474
    )


def saygili_and_rathje_08_eq5(ay, a_max, **kwargs):
    """Saygili and Rathje (2008) empirical correlation for uₚ.

    Eq. 20 in Meehan & Vahedifard (2013). Eq. 5 in Saygili and Rathje (2008).

    Parameters
    ----------
    ay : float
        Critical acceleration, in fraction of g.
    a_max : float
        Peak horizontal ground acceleration, in fraction of g.

    Returns
    -------
    perm_disp : float
        Permanent displacement, in [m].
    """
    # ay[np.where(ay > a_max)] = a_max  # There is no excess acceleration
    car = ay / a_max  # Critical acceleration ratio
    return 1e-2 * np.exp(
        5.52
        - 4.43 * car
        - 20.93 * car**2
        + 42.61 * car**3
        - 28.74 * car**4
        + 0.72 * np.log(a_max)
    )


def saygili_and_rathje_08_eq6col2(ay, a_max, v_max, **kwargs):
    """Saygili and Rathje (2008) empirical correlation for uₚ.

    Eq. 21 in Meehan & Vahedifard (2013). Eq. 6 in Saygili and Rathje (2008)
    with parameters from Table 1, column 2.

    Parameters
    ----------
    ay : float
        Critical acceleration, in fraction of g.
    a_max : float
        Peak horizontal ground acceleration, in fraction of g.
    v_max : float
        Peak horizontal ground velocity, in [cm/s].

    Returns
    -------
    perm_disp : float
        Permanent displacement, in [m].
    """
    # ay[np.where(ay > a_max)] = a_max  # There is no excess acceleration
    car = ay / a_max  # Critical acceleration ratio
    return 1e-2 * np.exp(
        -1.56
        - 4.58 * car
        - 20.84 * car**2
        + 44.75 * car**3
        - 30.5 * car**4
        - 0.64 * np.log(a_max)
        + 1.55 * np.log(v_max)
    )


def saygili_and_rathje_08_eq6col4(ay, a_max, arias_int, **kwargs):
    """Saygili and Rathje (2008) empirical correlation for uₚ.

    Not in Meehan & Vahedifard (2013), but it is equivalent to Eq. 21.
    Eq. 6 in Saygili and Rathje (2008) with parameters from Table 1, column 4.

    Parameters
    ----------
    ay : float
        Critical acceleration, in fraction of g.
    a_max : float
        Peak horizontal ground acceleration, in fraction of g.
    arias_int : float
        Arias intensity, in [m/s].

    Returns
    -------
    perm_disp : float
        Permanent displacement, in [m].
    """
    # ay[np.where(ay > a_max)] = a_max  # There is no excess acceleration
    car = ay / a_max  # Critical acceleration ratio
    return 1e-2 * np.exp(
        2.39
        - 5.24 * car
        - 18.78 * car**2
        + 42.01 * car**3
        - 29.15 * car**4
        - 1.56 * np.log(a_max)
        + 1.38 * np.log(arias_int)
    )


correlations = {  # based on numbering scheme by Meehan & Vahedifard (2013)
    "7": hynesgriffin_and_Franklin_1984_ub,
    "8": hynesgriffin_and_Franklin_1984_mean,
    "10": ambraseys_and_menu_88,
    "14": bray_and_travasarou_07_eq6,
    "16": jibson_07_eq6,
    "18": jibson_07_eq9,
    "19": jibson_07_eq10,
    "20": saygili_and_rathje_08_eq5,
    "21": saygili_and_rathje_08_eq6col2,
    "21_4": saygili_and_rathje_08_eq6col4,
}
