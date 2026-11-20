"""Core metrics to evaluate physical consistency of simulations and projections."""

# Adapted from https://github.com/Etienne-Meunier/Metrics-Ocean
# Original author: Etienne Meunier

import xarray

def density_horizontal_mean(density: xarray.DataArray, file_mask: xarray.Dataset) -> xarray.DataArray:
    """
    Compute the area weighted horizontal mean density at each depth
    
    Parameters
    ----------
    density: xarray.DataArray
    Density data with dimensions (time_counter, depth, y, x)
    file_mask: xarray.Dataset
    Dataset containing the grid mask variables (e1t, e2t, tmask)

    Returns
    -------
    xarray.DataArray
    2D DataArray over (time_counter, depth) representing the area weighted 
    horizontal mean density at each depth
    """
    e1t=file_mask.e1t.squeeze()
    e2t=file_mask.e2t.squeeze()
    tmask=file_mask.tmask.squeeze()
    area=e1t*e2t*tmask
    return (
        (density*area).sum(
            dim=["y", "x"], skipna=True,
        )
        /area.where(density.notnull()).sum(
        dim=["y", "x"],
        skipna=True
        )
    )

def temperature_horizontal_mean(temperature: xarray.DataArray, file_mask: xarray.Dataset) -> xarray.DataArray:
    """
    Compute the area weighted horizontal mean temperature at each depth
    
    Parameters
    ----------
    temperature: xarray.DataArray
    Temperature data with dimensions (time_counter, depth, y, x)
    file_mask: xarray.Dataset
    Dataset containing the grid mask variables (e1t, e2t, tmask)

    Returns
    -------
    xarray.DataArray
    2D DataArray over (time_counter, depth) representing the area weighted 
    horizontal mean temperature at each depth
    """
    e1t=file_mask.e1t.squeeze()
    e2t=file_mask.e2t.squeeze()
    tmask=file_mask.tmask.squeeze()
    area=e1t*e2t*tmask
    return (
        (temperature*area).sum(
            dim=["y", "x"], skipna=True,
        )
        /area.where(temperature.notnull()).sum(
        dim=["y", "x"],
        skipna=True
        )
    )

def salinity_horizontal_mean(salinity: xarray.DataArray, file_mask: xarray.Dataset) -> xarray.DataArray:
    """
    Compute the area weighted horizontal mean salinity at each depth
    
    Parameters
    ----------
    salinity: xarray.DataArray
    Salinity data with dimensions (time_counter, depth, y, x)
    file_mask: xarray.Dataset
    Dataset containing the grid mask variables (e1t, e2t, tmask)

    Returns
    -------
    xarray.DataArray
    2D DataArray over (time_counter, depth) representing the area weighted 
    horizontal mean salinity at each depth
    """
    e1t=file_mask.e1t.squeeze()
    e2t=file_mask.e2t.squeeze()
    tmask=file_mask.tmask.squeeze()
    area=e1t*e2t*tmask
    return (
        (salinity*area).sum(
            dim=["y", "x"], skipna=True,
        )
        /area.where(salinity.notnull()).sum(
        dim=["y", "x"],
        skipna=True
        )
    )
    
def density_delta(density: xarray.DataArray, file_mask: xarray.Dataset) -> xarray.DataArray:
    """ 
    Computes consecutive depth differences in the horizontal mean density
    The difference is defined as:
    rho(deeper)-rho(shallower)
    
    Positive values indicate increasing density with depth
    The returned depth coordinate corresponds to the shallower level of each consecutive depth pair

    Parameters
    ----------
    density: xarray.DataArray
    Density data with dimensions (time_counter, depth, y, x)
    file_mask: xarray.Dataset
    Dataset containing the grid mask variables (e1t, e2t, tmask)

    Returns
    -------
    xarray.DataArray
    2D DataArray over (time_counter, depth-1) representing the consecutive-depth density differences

    """
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    tmask = file_mask.tmask.squeeze()

    area = e1t*e2t*tmask

    density_horizontal = (
        (density*area).sum(
            dim=["y", "x"],
            skipna = True,
        )
        /area.where(
            density.notnull()
        ).sum(
            dim=["y", "x"],
            skipna = True
        )
    )
    
    delta=density_horizontal.diff(dim="depth")

    if "depth" in density_horizontal.coords:
        delta= delta.assign_coords(
            depth=density_horizontal["depth"].isel(
                depth=slice(0,-1)

            )
        )
    return delta

def density_violation_fraction_by_depth(density: xarray.DataArray, epsilon: float = 1e-5) -> xarray.DataArray:
    """
    Compute the fraction of horizontal grid cells violating the depth density monotonicity constraint at each consecutive depth pair.
    Violation occurs when density decreases with depth by more than epsilon
    
    Parameters
    ----------
    density : xarray.DataArray
    Density data with dimensions (time_counter, depth, y, x)

    epsilon : float
    Threshold used to determine monotonicity violation
    Default is 1e-5

    Returns
    -------
    xarray.DataArray
    2D DataArray over (time_counter, depth-1), representing the fraction of horizontal
    grid cells violating density monotonicity at each consecutive depth pair
  
    """

    density = density.where( density != 0)
    delta = density.diff(dim="depth")
    violation = (delta < -epsilon).where(delta.notnull())
    badprop_depth = violation.mean(
        dim=["y", "x"],
        skipna=True,
    )

    if "depth" in density.coords:
        badprop_depth = badprop_depth.assign_coords(
            depth=density["depth"].isel(
                depth=slice(0,-1)
            )
        )
    return badprop_depth

def density_500m(density: xarray.DataArray, file_mask: xarray.Dataset) -> xarray.DataArray:
    """ 
    Compute the volume-weighted mean density in the upper 500 m

    Parameters
    ----------
    density: xarray.DataArray
    Density data with dimensions (time_counter, depth, y, x)

    file_mask: xarray.Dataset
    Dataset containing the grid mask variables
    (e1t, e2t, e3t_0, tmask)

    Returns
    -------
    xarray.DataArray
    1D DataArray over time_counter, representing the volume-weighted mean density in the upper 500 m
    
    """
    depth_500 = 500
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask=file_mask.tmask.squeeze()
    
    density_500 = density.depth < depth_500
    density_region = density.where(density_500)
    vol_500 = e1t*e2t*e3t*tmask.where(density_500)

    return (
        (density_region*vol_500).sum(
            dim=["y", "x", "depth"],
            skipna=True
        )
        / vol_500.where(
            density_region.notnull()
        ).sum(
            dim=["y", "x", "depth"],
            skipna = True,
        )
    )

def density_500_3000m(density: xarray.DataArray, file_mask: xarray.Dataset) -> xarray.DataArray:
    """ 
    Computes the volume weighted mean density between 500 m and 3000 m

    Parameters
    ----------
    density : xarray.DataArray
    Density data with dimensions (time_counter, depth, y, x)
    
    file_mask : xarray.Dataset
    Dataset containing the grid_mask variables
    (e1t, e2t, e3t_0, tmask)

    Returns
    --------
    xarray.DataArray
    1D DataArray over time counter, representing the volume weighted mean density between
    500 m and 3000 m    
    """

    depth_min = 500
    depth_max = 3000

    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask = file_mask.tmask.squeeze()

    density_3000 = (
        (density.depth >= depth_min)
        & (density.depth < depth_max)
    )

    density_region = density.where(density_3000)
    vol_500_3000m = e1t*e2t*e3t*tmask.where(density_3000)

    return (
        ( density_region * vol_500_3000m).sum(
            dim=["y", "x", "depth"],
            skipna=True
        )
        /vol_500_3000m.where(
            density_region.notnull()
        ).sum(
            dim=["y", "x", "depth"],
            skipna=True
        )
    )

def density_3000m_bottom(density: xarray.DataArray, file_mask: xarray.Dataset) -> xarray.DataArray:
    """ 
    Computes volume-weighted mean density below 3000m
    
    Parameters
    ----------
    density : xarray.DataArray
    Density data with dimensions (time_counter, depth, y, x)

    file_mask: xarray.Dataset
    Dataset containing the grid mask variables
    (e1t, e2t, e3t_0, tmask)

    Returns
    -------
    xarray.DataArray
    1D DataArray over time_counter, representing the volume weighted mean density below 3000m
    """
    depth_min = 3000
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask=file_mask.tmask.squeeze()

    density_bottom = density.depth >= depth_min
    density_region = density.where(density_bottom)
    vol_3000m_bottom = e1t*e2t*e3t*tmask.where(density_bottom)

    return (
        (density_region * vol_3000m_bottom).sum(
            dim=["y", "x", "depth"],
            skipna=True,
        )
        /vol_3000m_bottom.where(
            density_region.notnull()
        ).sum(
            dim=["y", "x", "depth"],
            skipna=True,
        )
    )

def density_global(density: xarray.DataArray, file_mask: xarray.Dataset)-> xarray.DataArray:
    """ 
    Computes the global ocean volume-weighted mean density

    Parameters
    ----------
    density : xarray.DataArray
    Density data with dimensions (time_counter, depth, y, x)

    file_mask: xarray.Dataset
    Dataset containing the grid mask variables 
    (e1t, e2t, e3t_0, tmask)

    Returns 
    -------
    xarray.DataArray
    1D DataArray over time_counter, representing the global ocean volume-weighted density
     
    """
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask = file_mask.tmask.squeeze()

    volume = e1t*e2t*e3t*tmask

    return (
        (density*volume).sum(
            dim=["y", "x", "depth"],
            skipna =True,
        )
        / volume.where(
            density.notnull()
        ).sum(
            dim=["y", "x", "depth"],
            skipna=True
        )
    )


def check_density(density: xarray.DataArray, epsilon: float = 1e-5) -> xarray.DataArray:
    """
    Check density monotonicity violations in ocean data.

    Calculate the proportion of grid points at each time step where
    the density profile violates the monotonic-with-depth constraint.

    Parameters
    ----------
    density : xarray.DataArray
        A 4D DataArray with dimensions including 'time_counter', 'depth',
        'y', and 'x'. Represents the density field over time and space.
    epsilon : float, optional
        A small threshold used to determine significant non-monotonicity.
        Default is 1e-5.

    Returns
    -------
    xarray.DataArray
        1D DataArray (over time_counter), with values in [0,1], representing
        the proportion of grid points per time step where density increases
        with depth beyond the epsilon threshold.
    """
    density = density.where(density != 0)
    diff = density - density.shift(depth=-1)
    bad_prop = (diff > epsilon).mean(dim=["depth", "y", "x"])

    return bad_prop

def check_density_volume_weighted(
    density: xarray.DataArray,
    file_mask: xarray.Dataset,
    epsilon: float = 0.0,
) -> xarray.DataArray:
    """
    Compute the volume-weighted percentage of vertical density
    monotonicity violations at each time step.

    A violation occurs when density decreases with depth:
        density[k + 1] - density[k] < -epsilon

    Each adjacent-level comparison is weighted by the volume
    of the shallower cell:
        V = e1t * e2t * e3t_0 * tmask

    Parameters
    ----------
    density : xarray.DataArray
        4D density field with dimensions
        (time_counter, depth, y, x).

    file_mask : xarray.Dataset
        Mesh-mask dataset containing e1t, e2t, e3t_0, and tmask.

    epsilon : float, optional
        Tolerance for density inversions. The default, 0.0,
        follows the strict inequality in the paper's metric.

    Returns
    -------
    xarray.DataArray
        1D DataArray over time_counter, containing the
        volume-weighted percentage of density violations [0, 100].
    """

    # Grid-cell volume at T points
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask = file_mask.tmask.squeeze()

    volume = e1t * e2t * e3t * tmask

    # Treat zero density as invalid, consistent with check_density()
    density = density.where(density != 0)

    # Compare each level with the next deeper level
    density_deeper = density.shift(depth=-1)
    delta = density_deeper - density

    # A comparison is valid only if both levels contain valid
    # density values and the shallower cell is wet.
    valid = (
        density.notnull()
        & density_deeper.notnull()
        & (volume > 0)
    )

    # Identify inversions, excluding invalid comparisons
    violations = (delta < -epsilon).where(valid)

    # Volume-weighted numerator and denominator
    weighted_violations = (
        volume * violations.astype(float)
    ).sum(
        dim=["depth", "y", "x"],
        skipna=True,
    )

    valid_volume = volume.where(valid).sum(
        dim=["depth", "y", "x"],
        skipna=True,
    )

    # Return percentage, matching the scale of the paper's metric
    density_error = 100 * weighted_violations / valid_volume

    density_error.name = "density_error_volume_weighted"
    density_error.attrs["units"] = "%"
    density_error.attrs["epsilon"] = epsilon

    return density_error


def temperature_500m_30NS_metric(
    temperature: xarray.DataArray, file_mask: xarray.Dataset
) -> xarray.DataArray:
    """
    Metric Extraction Function.

    Compute the average Temperature at 500m depth between 30N and 30S. Unit: °C.

    Parameters
    ----------
    temperature : xarray.DataArray
        Temperature data with dimensions (time_counter, depth, y, x) and 2D spatial
        coordinates (nav_lat, nav_lon).
    file_mask   : xarray.Dataset
        Dataset containing the grid mask variables (e1t, e2t, e3t_0, tmask).

    Returns
    -------
    xarray.DataArray
        1D DataArray (over time_counter), representing the area-weighted mean
        temperature at 500m between 30N and 30S.
    """
    # Taking Temperature At 500m depth and between 30N and 30S.
    DEPTH = 500  # m : upper bound for deep water box
    LAT_BOUND = 30  # degrees N/S : tropical boundary

    t500_30NS = temperature.sel(depth=DEPTH, method="nearest").where(
        abs(temperature.nav_lat) < LAT_BOUND,
        drop=False,
    )
    # Computing Area Weights from Mask over 30N-30S latitude zone and @500m depth
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask = file_mask.tmask.squeeze()
    vol_500m_30NS = (
        e1t
        * e2t
        * e3t
        * tmask.sel(depth=DEPTH, method="nearest").where(
            abs(temperature.nav_lat) < LAT_BOUND,
            drop=False,
        )
    )
    # Returning Average Temperature at 500m depth as a numpy scalar
    return (t500_30NS * vol_500m_30NS).sum(dim=["y", "x", "depth"]) / vol_500m_30NS.sum(
        dim=["y", "x", "depth"]
    )


def temperature_BWbox_metric(
    temperature: xarray.DataArray, file_mask: xarray.Dataset
) -> xarray.DataArray:
    """
    Metric Extraction in a "Bottom Water" box.

    Average Temperature in a U-shaped "Bottom Water" box corresponding to waters below
    3000m or beyond 30 degrees of latitude North and South.

    .. code-block:: text

        ________________________________________________ _Surface
        | . . . . |__________________________| . . . . |_500m
        | . . . . |                          | . . . . |
        | . . . . |        Deep Water        | . . . . |
        | . . . . |__________________________| . . . . |_3000m
        | . . . . . . . . Bottom Water . . . . . . . . |
        |______________________________________________|_Bottom
        S        30S           Eq.          30N        N

    Figure: Schematic Representation of the Bottom Water box used in this metric.
    Unit: °C

    Parameters
    ----------
    temperature    : xarray.DataArray
        Temperature data with dimensions (time_counter, depth, y, x) and 2D spatial
        coordinates (nav_lat, nav_lon).
    file_mask : xarray.Dataset
        Dataset containing the grid mask variables (e1t, e2t, e3t_0, tmask).

    Returns
    -------
    xarray.DataArray
        1D DataArray (over time_counter), representing the area-weighted mean
        temperature in the Bottom Water box.
    """
    BD_DEPTH_MIN = 3000  # m : bottom water box lower bound
    LAT_BOUND = 30  # degrees N/S : tropical boundary

    condition = 1 - (temperature.depth < BD_DEPTH_MIN) * (
        abs(temperature.nav_lat) < LAT_BOUND
    )

    t_BW = temperature.where(condition)
    # Computing Area Weights from Mask over Box
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask = file_mask.tmask.squeeze()

    # Return Average Temperature on Box
    vol_BW = e1t * e2t * e3t * tmask.where(condition)
    return (t_BW * vol_BW).sum(dim=["y", "x", "depth"]) / vol_BW.sum(
        dim=["y", "x", "depth"]
    )


def temperature_DWbox_metric(
    temperature: xarray.DataArray, file_mask: xarray.Dataset
) -> xarray.DataArray:
    """
    Metric Extraction in a "Deep Water" box.

    Average Temperature in a "Deep Water" box corresponding to waters between 500m
    and 3000m depth and 30°N and 30°S.

    .. code-block:: text

        ________________________________________________ _Surface
        |         |__________________________|         |_500m
        |         | . . . . . . . . . . . . .|         |
        |         | . . . .Deep Water . . . .|         |
        |         |__________________________|         |_3000m
        |                 Bottom Water                 |
        |______________________________________________|_Bottom
        S        30S           Eq.          30N        N

    Figure: Schematic Representation of the Deep Water box used in this metric.
    Unit: °C

    Parameters
    ----------
    temperature    : xarray.DataArray
        Temperature data with dimensions (time_counter, depth, y, x) and 2D spatial
        coordinates (nav_lat, nav_lon).
    file_mask : xarray.Dataset
        Dataset containing the grid mask variables (e1t, e2t, e3t_0, tmask).

    Returns
    -------
    xarray.DataArray
        1D DataArray (over time_counter), representing the area-weighted mean
        temperature in the Deep Water box.
    """
    DW_DEPTH_MID = 1750  # m : deep water mid depth
    DW_DEPTH_HW = 1250  # m : half-width around mid depth
    LAT_BOUND = 30  # degrees N/S : tropical boundary

    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask = file_mask.tmask.squeeze()

    condition_HW = (abs(temperature.depth - DW_DEPTH_MID) < DW_DEPTH_HW) * (
        abs(temperature.nav_lat) < LAT_BOUND
    )

    t_DW = temperature.where(condition_HW)
    # Computing Volume Weights from Mask over box
    vol_DW = e1t * e2t * e3t * tmask.where(condition_HW)

    # Returning Average Temperature on box
    return (t_DW * vol_DW).sum(dim=["y", "x", "depth"]) / vol_DW.sum(
        dim=["y", "x", "depth"]
    )


def ACC_Drake_metric(
    uo: xarray.DataArray, file_mask: xarray.Dataset
) -> xarray.DataArray:
    """
    Metric Extraction in the Drake Passage.

    Antarctic Circumpolar Current Transport at the DINO equivalent of the Drake Passage
    at (x=0).

    Version 1 of ACC metric: Computes the flux assuming rigid lid (no sea surface height
    variations), thus using the original e3u_0 variable from the mask.
    Unit: Sv

    Parameters
    ----------
    uo        : xarray.DataArray  
        Zonal velocity data for each point with dimension (t, depth, y, x)
        and 2D spatial coordinates (nav_lat, nav_lon).
    file_mask : xarray.Dataset
        Dataset containing the grid mask variables (e2u, e3u_0, umask).

    Returns
    -------
    xarray.DataArray
        1D DataArray (over time_counter), representing the total transport across Drake
        Passage (in Sv).
    """
    umask_Drake = file_mask.umask.isel(x=0).squeeze()
    e3u = file_mask.e3u_0.squeeze()
    e2u = file_mask.e2u.squeeze()

    # Masking the variables onto the Drake Passage

    u_masked = uo.isel(x=0) * umask_Drake
    e3u_masked = e3u.isel(x=0) * umask_Drake
    e2u_masked = e2u.isel(x=0) * umask_Drake

    # Multiplying zonal velocity by the sectional areas (e2u*e3u)

    ubar = u_masked * e3u_masked
    flux = (e2u_masked * ubar).sum(dim=["y", "depth"])
    # Returning Total Transport across Drake passage as a numpy scalar (unit: Sv)
    return flux / 1e6


def ACC_Drake_metric_2(
    uo: xarray.DataArray, ssh: xarray.DataArray, file_mask: xarray.Dataset
) -> xarray.DataArray:
    """
    Metric Extraction of the Drake Passage.

    Antarctic Circumpolar Current Transport at the DINO equivalent of the Drake Passage
    at (x=0).
    Version 2 of ACC metric: Computes the flux assuming varying ssh, thus needing to
    recompute e3u variable from e3u_0.

    Unit: Sv

    Parameters
    ----------
    uo        : xarray.DataArray
        Zonal velocity data with dimension (t, depth, y, x) and spatial coordinates
        (nav_lat, nav_lon). Zonal velocity value for each point.
    ssh       : xarray.DataArray
        Sea Surface Height data (t, y, x) and spatial coordinates (nav_lat, nav_lon).
    file_mask : xarray.Dataset
        Dataset containing the grid mask variables (e2u, e3u_0, umask).

    Returns
    -------
    xarray.DataArray
        1D DataArray (over time_counter), representing the total transport across Drake
        Passage (in Sv).
    """
    umask_Drake = file_mask.umask.isel(x=0).squeeze()
    e3u_0 = file_mask.e3u_0.squeeze()
    e2u = file_mask.e2u.squeeze()

    # Recomputing e3u, using ssh to refactor the original e3u_0 cell heights)
    ssh_u = (ssh + ssh.roll(x=-1)) / 2
    bathy_u = e3u_0.sum(dim="depth")
    ssumask = umask_Drake[:, 0]
    e3u = e3u_0 * (1 + ssh_u * ssumask / (bathy_u + 1 - ssumask))
    # Masking the variables onto the Drake Passage
    u_masked = uo.isel(x=0) * umask_Drake
    e3u_masked = e3u.isel(x=0) * umask_Drake
    e2u_masked = e2u.isel(x=0) * umask_Drake
    # Multiplying zonal velocity by the sectional areas (e2u*e3u)
    ubar = u_masked * e3u_masked
    flux = (e2u_masked * ubar).sum(dim=["y", "depth"])
    # Return Total Transport across Drake passage as a numpy scalar (unit: Sv)
    return flux / 1e6


def NASTG_BSF_max(
    vo: xarray.DataArray, ssh: xarray.DataArray, file_mask: xarray.Dataset
) -> xarray.DataArray:
    """
    Metric Extraction of the North-Atlantic SubTropical Gyre (NASTG).

    Intensity of the North-Atlantic SubTropical Gyre (NASTG) containing the Gulf-Stream
    current computed from the local maximum of the Barotropic Stream Function (BSF)
    Unit: Sv

    Parameters
    ----------
    vo        : xarray.DataArray
        Meridional velocity data with dimension (t, depth, y, x) and coordinate
        (nav_lat, nav_lon). Meridional velocity value for each point.
    ssh       : xarray.DataArray
        Sea Surface Height data with dimensions (t, y, x) and spatial coordinates
        (nav_lat, nav_lon).
    file_mask : xarray.Dataset
        Dataset containing the grid mask variables (e1v, e2v, vmask).

    Returns
    -------
    xarray.DataArray
        1D DataArray (over time_counter), representing the maximum Barotropic Stream
        Function in the NASTG region (in Sv).
    """
    e3v_0 = file_mask.e3v_0.squeeze()
    e1v = file_mask.e1v.squeeze()
    vmask = file_mask.vmask.squeeze()
    # Updating e3v from e3v_0 and SSH
    ssh_v = (ssh + ssh.roll(y=-1)) / 2
    bathy_v = e3v_0.sum(dim="depth")
    ssvmask = vmask.isel(depth=0)
    e3v = e3v_0 * (1 + ssh_v * ssvmask / (bathy_v + 1 - ssvmask))

    # Integrating Meridional Transport (e3v*e1v*vo) along depth and X from Western
    # boundary eastward
    # to get Barotropic Stream Function with the "American continent" as reference point
    # (BSF=0)
    V = (vo * e3v).sum(dim="depth")  #  == "Barotropic Velocity" * Bathymetry
    BSF = (V * e1v * ssvmask).cumsum(
        dim="x"
    ) / 1e6  # Integrating from the West, and converting from m³/s to Sv
    # Selecting 0N-40N window where to search for the maximum, which will correspond to
    # the center of rotation for the gyre
    BSF_NASPG = BSF.where(abs(BSF.nav_lat - 20) < 20)  # noqa: PLR2004

    # Selecting the maximum value of the BSF in the selected window
    # and return it as a numpy scalar
    return BSF_NASPG.max(dim=["y", "x"])


# ---------------------------------------------------------------------------
# Pointwise fields (for cell-by-cell RMSE / MAE)
# ---------------------------------------------------------------------------


def _depth_dim(da: xarray.DataArray):
    for cand in ("depth", "nav_lev", "deptht"):
        if cand in da.dims:
            return cand
    return None


def pointwise_temperature(temperature, file_mask):
    """Return temperature (t, depth, y, x) masked to ocean T-cells."""
    tmask = file_mask.tmask.squeeze()
    return temperature.where(tmask == 1)


def pointwise_salinity(salinity, file_mask):
    """Return salinity (t, depth, y, x) masked to ocean T-cells."""
    tmask = file_mask.tmask.squeeze()
    return salinity.where(tmask == 1)


def pointwise_density(density, file_mask):
    """Return density (t, depth, y, x) masked to ocean T-cells."""
    tmask = file_mask.tmask.squeeze()
    return density.where(tmask == 1)


def pointwise_ssh(ssh, file_mask):
    """Return SSH (t, y, x) masked to ocean surface cells."""
    tmask = file_mask.tmask.squeeze()
    dd = _depth_dim(tmask)
    if dd is not None:
        smask = tmask.isel({dd: 0})
    else:
        smask = tmask
    return ssh.where(smask == 1)


def pointwise_velocity_u(velocity_u, file_mask):
    """Return zonal velocity (t, depth, y, x) masked to ocean U-cells."""
    umask = file_mask.umask.squeeze()
    return velocity_u.where(umask == 1)


def pointwise_velocity_v(velocity_v, file_mask):
    """Return meridional velocity (t, depth, y, x) masked to ocean V-cells."""
    vmask = file_mask.vmask.squeeze()
    return velocity_v.where(vmask == 1)
