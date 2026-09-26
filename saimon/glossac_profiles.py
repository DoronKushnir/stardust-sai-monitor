"""
glossac_profiles.py -- Vertical extinction profiling from GloSSAC dataset.

Provides functionality to extract area-weighted, time-averaged tropical 
stratospheric vertical extinction profiles for specific epochs.
"""

import numpy as np
from .glossac import GloSSACLoader


def extract_epoch_profile(
    wavelength_nm: int,
    start_yyyymm: int,
    end_yyyymm: int,
    lat_range: tuple = (-20, 20)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts the area-weighted, time-averaged vertical extinction profile
    from GloSSAC for a specified wavelength and time window.

    Parameters
    ----------
    wavelength_nm : int
        Wavelength in nm (e.g., 525, 1020).
    start_yyyymm : int
        Start of the time window (inclusive).
    end_yyyymm : int
        End of the time window (inclusive).
    lat_range : tuple
        Latitude range (min_lat, max_lat) for tropical average.

    Returns
    -------
    alt_km : np.ndarray
        Altitude grid in km.
    ext_mean : np.ndarray
        Time- and area-averaged extinction profile [m^-1].
    """
    loader = GloSSACLoader()
    data = loader.load_raw_data(wavelength_nm=wavelength_nm)
    
    ext = data['ext']  # (time, lat, alt)
    lat = data['lat']
    alt_km = data['alt']
    time = data['time']

    # Time mask
    time_mask = (time >= start_yyyymm) & (time <= end_yyyymm)
    
    # Latitude mask
    lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
    tropical_lat = lat[lat_mask]
    
    # Slice data: shape (time, lat, alt)
    ext_slice = ext[time_mask][:, lat_mask, :]
    
    # Weights based on latitude
    weights = np.cos(np.deg2rad(tropical_lat))
    weights_2d = weights[np.newaxis, :, np.newaxis]
    
    # We want to average over time and latitude, but taking care of masked values (-999.0).
    # Since ext is already a masked array, we can use np.ma.average.
    # We will average over time (axis=0) and latitude (axis=1).
    # We can flatten time and lat into one dimension and apply corresponding weights.
    
    nt, nlat, nalt = ext_slice.shape
    
    # Reshape weights to match time x lat
    w = np.tile(weights, (nt, 1))
    
    # Flatten time and lat
    ext_flat = ext_slice.reshape(nt * nlat, nalt)
    w_flat = w.reshape(nt * nlat)
    
    # Calculate weighted average over the first axis (time * lat), ignoring masked values.
    # Convert ext_flat to masked array explicitly to be safe
    ext_ma = np.ma.masked_invalid(ext_flat)
    
    ext_mean = np.zeros(nalt)
    for k in range(nalt):
        valid = ~ext_ma[:, k].mask
        if np.any(valid):
            ext_mean[k] = np.average(ext_ma[valid, k], weights=w_flat[valid])
        else:
            ext_mean[k] = np.nan
            
    # Convert GloSSAC extinction from km^-1 to m^-1
    ext_mean_m1 = ext_mean / 1000.0

    return alt_km, ext_mean_m1
