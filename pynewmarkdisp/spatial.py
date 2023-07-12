# ``pynewmarkdisp/spatial.py``

"""Functions to calculate permanent displacements in a spatial domain."""

from itertools import product
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numba import njit

from pynewmarkdisp.newmark import direct_newmark
from pynewmarkdisp.infslope import factor_of_safety, get_ky, num_to_2darray

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


def load_ascii_raster(path, dtype=None):
    """Load a raster file in ASCII format (`.ASC` extension).

    Parameters
    ----------
    path : str
        File to be loaded. It must be in [Esri ASCII raster format](
        https://en.wikipedia.org/wiki/Esri_grid).
        Its header must be 6 lines long, and the spatial location of the raster
        is specified by the location of the lower left corner of the lower
        left cell, as shown below:
        ```
            NCOLS xxx
            NROWS xxx
            XLLCORNER xxx
            YLLCORNER xxx
            CELLSIZE xxx
            NODATA_VALUE xxx
            row 1
            row 2
            ...
            row n
        ```
    dtype : data-type, optional
        Data type of the returned array. If not given (i.e., `dtype=None`), the
        data type is determined by the contents of the file. Its default value
        is `None`.

    Returns
    -------
    raster : (m, n) ndarray
        Raster data given as a 2D array.
    header : dict
        Header of the raster file.
    """
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
    raster[np.where(raster == header["nodata_value"])] = np.nan
    raster = raster.astype(dtype) if dtype is not None else raster
    return (raster, header)


def map_zones(parameters, zones):
    """Associate geotechnical parameters to zones.

    Parameters
    ----------
    parameters : dict
        Dictionary with the geotechnical parameters of each zone. The keys are
        the zone numbers, and the values are tuples with the zone's friction
        angle in [°], cohesion in [kPa], and unit weight in [kN/m³],
        respectively.
    zones : (m, n) ndarray
        2D array with the zone number assigned to each cell. There has to be as
        many zone numbers as keys in the parameters dictionary.

    Returns
    -------
    phi, c, gamma : (m, n) ndarray, (m, n) ndarray, (m, n) ndarray
        Three 2D arrays with the spatial distribution of the friction angle,
        cohesion, and unit weight, respectively. Each array has the same shape
        as ``zones``.
    """
    phi = np.empty_like(zones)
    c = np.empty_like(zones)
    gamma = np.empty_like(zones)
    for z in parameters.keys():  # Zone: frict_angle, cohesion, gamma
        phi[np.where(zones == z)] = parameters[z][0]
        c[np.where(zones == z)] = parameters[z][1]
        gamma[np.where(zones == z)] = parameters[z][2]
    return phi, c, gamma


@njit(fastmath=True, cache=True)
def hzt_accel_from_2_dir(accel_ew, accel_ns, azimuth):
    """Calculate the resultant acceleration from two horizontal components.

    This funtion is optimized for speed using Numba.

    Parameters
    ----------
    accel_ew : (n,) ndarray
        1D array with the acceleration series in the East-West direction.
    accel_ns : (n,) ndarray
        1D array with the acceleration series in the North-South direction.
    azimuth : float or int
        Azimuth of the slope dip direction, in [°] and measured clockwise from
        North.

    Returns
    -------
    accel_hzt : (n, ) ndarray
        1D array with the resultant horizontal acceleration series.
    """
    azimuth_r = np.radians(azimuth)
    return accel_ew * np.sin(azimuth_r) + accel_ns * np.cos(azimuth_r)


def spatial_hzt_accel_from_2_dir(accel_ew, accel_ns, azimuth):
    """Calculate spatially the resultant acceleration from two hztl components.

    The azimuth is spatially distributed (e.g., obtained using the
    ``get_dip_azimuth`` function), but ``accel_ew``, ``accel_ns``, and
    ``time_vect`` are constant over all the spatial domain.

    Parameters
    ----------
    accel_ew : (p,) ndarray
        1D array with the acceleration series in the East-West direction.
    accel_ns : (p,) ndarray
        1D array with the acceleration series in the North-South direction.
    azimuth : (m, n) ndarray
        2D array with the spatial distribution of the azimuth of the slope dip
        direction, in [°] and measured clockwise from North.

    Returns
    -------
    accel_hzt : (m, n) ndarray
        2D array with the resultant horizontal acceleration series calculated
        at each cell. There is a 1D array with the resultant horizontal
        acceleration series at each cell (i, j).
    """
    row, col = azimuth.shape
    azimuth_r = np.radians(azimuth)
    accel_hzt = np.empty_like(azimuth_r, dtype=object)
    for i, j in product(range(row), range(col)):
        accel_hzt[i, j] = hzt_accel_from_2_dir(
            accel_ew, accel_ns, azimuth[i, j]
        )
    return accel_hzt


def _select_case(cell, time, accel, azimuth=None):
    """Select the proper case for a cell depending on the spatial inputs.

    It is used by the ``spatial_newmark`` function and checks if the
    acceleration is bidirectional or unidirectional, and if the azimuth is
    constant or spatially distributed. It returns the proper acceleration and
    time vectors for the cell (i, j).

    Parameters
    ----------
    cell : tuple
        Tuple with the cell coordinates (i, j).
    time : ndarray
        1D or 2D ndarray with the time data.
    accel : ndarray
        1D or 2D ndarray with the acceleration data.
    azimuth : (m, n) ndarray or None, optional
        2D array with the spatial distribution of the azimuth of the slope dip
        direction, in [°] and measured clockwise from North. By default its
        value is ``None``.

    Returns
    -------
    time : ndarray
        1D array with the time series for the cell (i, j).
    accel : ndarray
        1D array with the acceleration series for the cell (i, j).
    """
    # Checking acceleration input
    if azimuth is not None:  # Bidirectional acceleration, modified by slope
        accel_ew, accel_ns = accel[0], accel[1]
        if accel[0].ndim == 2:  # Bidirect. acceleration varies at all (i, j)
            accel_ew, accel_ns = accel_ew[cell], accel_ns[cell]
        #  else:  # Bidirectional acceleration constant over all (i, j)
        accel = hzt_accel_from_2_dir(accel_ew, accel_ns, azimuth[cell])
    elif accel.ndim == 2:  # Unidirectional acceleration varies at all (i, j)
        accel = accel[cell]
    else:  # Unidirectional acceleration constantant over all (i, j)
        accel = accel.copy()
    # Checking time input: Time vector varies at all (i, j) or is constant
    time = time[cell] if time.ndim == 2 else time.copy()
    return time, accel


def spatial_newmark(time, accel, ky, g, azimuth=None):
    """Calculate the spatial distribution of permanent displacements.

    Perform the direct Newmarkl method for each cell of the spatial domain and
    return the spatial distribution of the permanent displacements.

    Parameters
    ----------
    time : ndarray
        1D or 2D array with the time data, in [s].
    accel : tuple, list or ndarray
        Object with the acceleration data. If it ``azimuth`` is given,
        ``accel`` must have a length of 2 and contain the acceleration in the
        East-West and North-South directions, respectively; both components of
        the acceleration might be constant or spatially distributed. If
        ``azimuth`` is ``None``, the ``accel`` vector might be spatially
        distributed (2D array) or constant (1D array).
    ky : ndarray
        2D array with the spatial distribution of the critical seismic
        coefficient.
    g : float
        Acceleration of gravity, in the same units as ``accel``. For example,
        if ``accel`` is in [m/s²], ``g`` should be equal to 9.81; if ``accel``
        is as a fraction of gravity, ``g`` should be equal to 1.
    azimuth : (m, n) ndarray or None, optional
        2D array with the spatial distribution of the azimuth of the slope dip
        direction, in [°] and measured clockwise from North. By default its
        value is ``None``.

    Returns
    -------
    permanent_disp : (m, n) ndarray
        2D array with the spatial distribution of permanent displacements.
    """
    ky = num_to_2darray(ky)
    row, col = ky.shape
    permanent_disp = np.full(ky.shape, np.nan)
    for i, j in product(range(row), range(col)):
        if np.isnan(ky[i, j]):
            continue  # Skip step and continue the loop as u_p=np.nan
        time_s, accel_s = _select_case((i, j), time, accel, azimuth)
        newmark_str = direct_newmark(time_s, accel_s, ky[i, j], g)
        permanent_disp[i, j] = newmark_str["perm_disp"]
    return np.around(permanent_disp, 3)


def verify_newmark_at_cell(
    cell, time, accel, g, depth, depth_w, slope, phi, c, gamma, azimuth=None
):
    """Verify the direct Newmark method at a specific cell.

    Parameters
    ----------
    cell : tuple
        Tuple with the index of the cell to be verified.
    time : ndarray
        1D or 2D array with the time data, in [s].
    accel : tuple or list or ndarray
        Object with the acceleration data. If it ``azimuth`` is given,
        ``accel`` must have a length of 2 and contain the acceleration in the
        East-West and North-South directions, respectively; both components of
        the acceleration might be constant or spatially distributed. If
        ``azimuth`` is ``None``, the ``accel`` vector might be spatially
        distributed (2D array) or constant (1D array).
    g : float
        Acceleration of gravity, in the same units as ``accel``. For example,
        if ``accel`` is in [m/s²], ``g`` should be equal to 9.81; if ``accel``
        is as a fraction of gravity, ``g`` should be equal to 1.
    depth : (m, n) ndarray
        2D array with the spatial distribution of the potential sliding surface
        depth, in [m].
    depth_w : ndarray
        2D array with the spatial distribution of the water table depth, in
        [m].
    slope : (m, n) ndarray
        2D array with the spatial distribution of the slope inclination, in
        [°].
    phi : (m, n) ndarray
        2D array with the spatial distribution of the internal friction angle,
        in [°].
    c : (m, n) ndarray
        2D array with the spatial distribution of the cohesion, in [kPa].
    gamma : ndarray
        2D array with the spatial distribution of the soil unit weight, in
        [kN/m³].
    azimuth : (m, n) ndarray or None, optional
        2D array with the spatial distribution of the azimuth of the slope dip
        direction, in [°] and measured clockwise from North. By default its
        value is ``None``.

    Returns
    -------
    newmark_str : dict
        Dictionary with the results of the direct Newmark method at the cell.
    """
    i, j = cell
    depth, depth_w, slope = depth[i, j], depth_w[i, j], slope[i, j]
    phi, c, gamma = phi[i, j], c[i, j], gamma[i, j]
    fs_static = factor_of_safety(depth, depth_w, slope, phi, c, gamma, ks=0)
    ky = get_ky(depth, depth_w, slope, phi, c, gamma)
    fs_ky = factor_of_safety(depth, depth_w, slope, phi, c, gamma, ky)
    time_s, accel_s = _select_case((i, j), time, accel, azimuth)
    newmark_str = direct_newmark(time_s, accel_s, ky, g)
    newmark_str["fs_static"] = fs_static
    newmark_str["fs_ky"] = fs_ky
    return newmark_str


def get_idx_at_coords(x, y, spat_ref):
    """Convert coordinates to array indexes.

    Parameters
    ----------
    x : int or float
        Coordinate in the x-direction.
    y : int or float
        Coordinate in the y-direction.
    spat_ref : dict
        Dictionary with the spatial reference of the raster. It contains the
        same information of the ASCII file header. See the documentation of
        the ``read_ascii`` function for more information.

    Returns
    -------
    indexes : tuple
        Tuple with the indexes of the array corresponding to the given
        coordinates.
    """
    i = int(
        spat_ref["nrows"]
        - np.ceil((y - spat_ref["yllcorner"]) / spat_ref["cellsize"])
    )
    j = int(np.floor((x - spat_ref["xllcorner"]) / spat_ref["cellsize"]))
    return i, j


def plot_spatial_field(field, dem=None, discrete=False, **kwargs):
    """
    Plot a spatial field.

    Parameters
    ----------
    field : (m, n) ndarray
        2D array with the spatial field to be plotted.
    dem : (m, n) ndarray or None, optional
        2D array with the digital elevation model (DEM) of the area. It is used
        to plot the hillshade below the spatial field to enhance the
        visualization. By default its value is ``None``.
    discrete : bool, optional
        If ``True``, the spatial field is plotted as a discrete field. By
        default its value is ``False``.
    **kwargs : dict, optional
        Additional keyword arguments. ``spat_ref`` is a dictionary with the
        spatial reference of the raster. It contains the same information of
        the ASCII file header. See the documentation of the ``read_ascii``
        function for more information. ``levels`` is a list or array with the
        contour levels. ``cmap`` is a colormap palette accepted by
        Matplotlib, or a ``str`` with the name of a Matplotlib colormap.
        ``label`` is a list of ``str`` with the labels of the unique elements
        in a discrete field. ``vmin`` and ``vmax`` define the data range that
        the colormap covers. ``labelrot`` is the rotation angle of the colorbar
        ticks labels. ``title`` is the title of the spatial field.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib object which might be used to save the figure as a file.
    """
    m, n = field.shape
    alpha = 1
    spat_ref = kwargs.get(
        "spat_ref", {"xllcorner": 0, "yllcorner": 0, "cellsize": 1}
    )
    extent = (
        spat_ref["xllcorner"],  # x min
        spat_ref["xllcorner"] + spat_ref["cellsize"] * n,  # x max
        spat_ref["yllcorner"],  # y min
        spat_ref["yllcorner"] + spat_ref["cellsize"] * m,  # y max
    )

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[5.5, 4.5], sharex=True)
    if dem is not None:  # Generate hillshade in background and contours
        alpha = 0.8  # Transparency for ploting the field over the hillshade
        hsd = get_hillshade(dem, cellsize=spat_ref["cellsize"])
        ax.matshow(hsd, cmap="gray", extent=extent)
        # ax.matshow(dem, cmap="gray", extent=extent, alpha=alpha)
        levels = kwargs.get("levels")
        cs = ax.contour(
            dem,
            colors="k",
            origin="upper",
            linewidths=0.5,
            extent=extent,
            levels=levels,
            alpha=0.8,
        )
        ax.clabel(cs, inline=True, fontsize="x-small")

    if discrete:  # Plot field as a discrete variable with unique values
        ticks = np.unique(field)[~np.isnan(np.unique(field))]  # For colorbar
        n = len(ticks)
        field_r = np.full_like(field, np.nan)  # Empty reclasified field
        ticks_r = np.arange(n)  # Ticks to use in reclasified field
        for i in ticks_r:  #  Fill reclasified field
            field_r[field == ticks[i]] = i
        cmap = plt.colormaps.get_cmap(kwargs.get("cmap")).resampled(n)  # type: ignore
        im = ax.matshow(
            field_r,  # plots the reclasified field instead of the original
            cmap=cmap,
            extent=extent,
            alpha=alpha,
            vmin=-0.5,
            vmax=n - 0.5,
        )  # map reclasified, but use the original labels
        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.01, ticks=ticks_r)
        ticks = np.round(ticks, 1) if ticks.dtype == np.float64 else ticks
        label = kwargs.get("label", ticks)
        _ = cbar.ax.set_yticklabels(label)
    else:  # Plot field as a continue variable
        cmap = plt.colormaps.get_cmap(kwargs.get("cmap"))  # type: ignore
        vmin = kwargs.get("vmin", np.nanmin(field))
        vmax = kwargs.get("vmax", np.nanmax(field))

        field_isfin = field[np.isfinite(field)]
        if np.nanmin(field_isfin) < vmin and np.nanmax(field_isfin) > vmax:
            extend = "both"
        elif np.nanmin(field_isfin) < vmin:
            extend = "min"
        elif np.nanmax(field_isfin) > vmax:
            extend = "max"
        else:
            extend = "neither"
        im = ax.matshow(
            field,
            cmap=cmap,
            extent=extent,
            alpha=alpha,
            vmin=kwargs.get("vmin"),
            vmax=kwargs.get("vmax"),
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.01, extend=extend)

    # Colorbar settings
    lr = kwargs.get("labelrot", 0)
    cbar.ax.tick_params(axis="y", labelrotation=lr, labelsize="small", pad=1.5)
    cbar.set_label(kwargs.get("title", "Scale"), rotation=90, size="large")
    [label.set_va("center") for label in cbar.ax.get_yticklabels()]
    # Axis settings
    ax.xaxis.set_tick_params(labelrotation=0, labelsize="small")
    ax.yaxis.set_tick_params(labelrotation=90, labelsize="small")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    [label.set_va("center") for label in ax.get_yticklabels()]
    for spine in ["bottom", "top", "left", "right"]:
        ax.spines[spine].set_linewidth(1.5)
    ax.grid(True, which="major", linestyle="--")
    fig.tight_layout()
    return fig


def get_dip_azimuth(dem, cellsize=1):
    """Calculate the azimuth of the slope dip direction from the North.

    Positive values are clockwise from the North. The azimuth is returned in
    [°], ranging from 0 to 360.

    Parameters
    ----------
    dem : (m, n) array
        2D array with the digital elevation model (DEM) of the area.
    cellsize : int, optional
        Cell size of the raster representing the DEM. Its default value is 1.

    Returns
    -------
    azimuth : (m, n) array
        Spatial distribution of the azimuth of the slope dip direction.
    """
    dzdy, dzdx = np.gradient(dem, cellsize)
    dzdx *= -1  # Adequate sign for the x component
    return np.degrees(0.5 * np.pi - np.arctan2(dzdy, dzdx)) % 360


def get_slope(dem, cellsize=1):
    """Calculate the slope (or inclination) of the terrain.

    Parameters
    ----------
    dem : (m, n) array
        2D array with the digital elevation model (DEM) of the area.
    cellsize : int, optional
        Cell size of the raster representing the DEM. Units must be consistent
        with the units of ``dem``. Its default value is 1.

    Returns
    -------
    slope : (m, n) array
        Spatial distribution of the slope, in degrees.
    """
    dzdy, dzdx = np.gradient(dem, cellsize)
    dzdx *= -1  # Adequate sign for the x component
    return np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))


def get_hillshade(dem, sun_azimuth=315, sun_altitude=30, cellsize=1):
    """Generate a hillshade image from the DEM.

    Parameters
    ----------
    dem : (m, n) array
        2D array with the digital elevation model (DEM) of the area
    sun_azimuth : int or float, optional
        Azimuth of the sun, in [°]. Its default value is 315.
    sun_altitude : int or float, optional
        Angle from the horizontal plane to the sun, in degrees, indicating the
        altitude of the sun. Its default value is 30.
    cellsize : int, optional
        Cell size of the raster representing the DEM. Units must be consistent
        with the units of ``dem``. Its default value is 1.

    Returns
    -------
    hillshade : (m, n) array
        2D array with the hillshade raster.
    """
    sun_azimuth_r = np.radians(sun_azimuth)
    sun_altitude_r = np.radians(sun_altitude)
    slope_r = np.radians(get_slope(dem, cellsize))
    azimuth_r = np.radians(get_dip_azimuth(dem, cellsize))
    # Lambertian reflectance
    term1 = (
        np.cos(azimuth_r - sun_azimuth_r)
        * np.sin(slope_r)
        * np.sin(0.5 * np.pi - sun_altitude_r)
    )
    term2 = np.cos(slope_r) * np.cos(0.5 * np.pi - sun_altitude_r)
    reflectance = term1 + term2
    ptp = np.nanmax(reflectance) - np.nanmin(reflectance)
    return 255 * (reflectance - np.nanmin(reflectance)) / ptp


def reclass_raster(field, limits):
    """Reclassify a raster based on a list of limits.

    Convert a continuous raster into a categorical one. The limits are used to
    define the categories.

    Parameters
    ----------
    field : (m, n) ndarray
        2D array with the spatial field to be reclassified.
    limits : array_like
        Limits to be used in the reclassification. The minimum and maximum
        values of the field should *not* be included as the first and last
        elements of the list, as they are automatically included in the limits.

    Returns
    -------
    field_reclass : (m, n) ndarray
        2D array with the reclassified field.
    """
    limits = np.hstack((np.nanmin(field), limits, np.nanmax(field)))
    field_reclass = (len(limits) - 1) * np.ones_like(field, dtype=int)
    for i, lim in enumerate(limits[:-1]):
        field_reclass[
            np.logical_and(field >= limits[i], field <= limits[i + 1])
        ] = i
    return field_reclass
