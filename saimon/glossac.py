import netCDF4 as nc
import numpy as np
import os

class GloSSACLoader:
    """Loader for GloSSAC V2.23 NetCDF files."""

    def __init__(self, file_path=None):
        from .config import GLOSSAC_NC
        file_path = str(GLOSSAC_NC) if file_path is None else file_path
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GloSSAC file not found at {file_path}")

    def load_raw_data(self, wavelength_nm=525):
        """Loads extinction, lat, alt, time, tropopause, and file-calculated OD.

        Parameters
        ----------
        wavelength_nm : int
            Wavelength in nm (386, 452, 525, 1020 are available in glossac).

        Returns
        -------
        dict
            Dictionary containing numpy arrays for 'ext', 'lat', 'alt', 'time', 'trp', 'od_file', 'wavelength'.
        """
        ds = nc.Dataset(self.file_path, 'r')
        try:
            wls = ds.variables['wavelengths_glossac'][:]
            wl_idx = np.argmin(np.abs(wls - wavelength_nm))
            actual_wl = wls[wl_idx]

            ext = ds.variables['Glossac_Aerosol_Extinction_Coefficient'][:, :, :, wl_idx]
            ext = np.ma.masked_equal(ext, -999.0)

            lat = ds.variables['lat'][:]
            alt = ds.variables['alt'][:]
            time = ds.variables['time'][:]
            trp = ds.variables['trp_hgt'][:]  # (lat, month)

            od_file = ds.variables['Glossac_Aerosol_Optical_Depth'][:, :, wl_idx]
            od_file = np.ma.masked_equal(od_file, -999.0)

            return {
                'ext': ext,
                'lat': lat,
                'alt': alt,
                'time': time,
                'trp': trp,
                'od_file': od_file,
                'wavelength': actual_wl
            }
        finally:
            ds.close()

def compute_tropical_saod(data, lat_range=(-20, 20), alt_max=40.0):
    """Computes tropical stratospheric SAOD time series.

    Integrates extinction from the tropopause to alt_max for each latitude bin
    within lat_range, then averages with cos(latitude) weighting.

    Parameters
    ----------
    data : dict
        Output from GloSSACLoader.load_raw_data().
    lat_range : tuple
        Latitude range (min_lat, max_lat) for tropical average.
    alt_max : float
        Upper altitude for integration (km).

    Returns
    -------
    time : np.ndarray
        Time in YYYYMM format.
    saod : np.ndarray
        Computed SAOD time series.
    saod_file : np.ndarray
        SAOD time series from the file's precomputed OD.
    """
    ext = data['ext']
    lat = data['lat']
    alt = data['alt']
    time = data['time']
    trp = data['trp']
    od_file = data['od_file']

    lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
    tropical_lat = lat[lat_mask]
    ext_trop = ext[:, lat_mask, :]
    trp_trop = trp[lat_mask, :]
    od_file_trop = od_file[:, lat_mask]

    nt, nlat, nalt = ext_trop.shape
    saod_comp = np.zeros(nt)
    saod_file = np.zeros(nt)

    weights = np.cos(np.deg2rad(tropical_lat))
    weights /= np.sum(weights)

    # dz is constant 0.5 km in GloSSAC
    dz = 0.5 

    for t_idx in range(nt):
        curr_time = time[t_idx]
        month_idx = (int(curr_time) % 100) - 1
        
        lat_saod_comp = []
        lat_saod_file = []
        
        for l_idx in range(nlat):
            # Computed SAOD
            curr_trp = trp_trop[l_idx, month_idx]
            alt_mask = (alt >= curr_trp) & (alt <= alt_max)
            
            curr_ext = ext_trop[t_idx, l_idx, alt_mask]
            if curr_ext.count() > 0:
                # Basic integration: midpoint or left-point?
                # alt is 0.5, 1.0, 1.5...
                # Each bin represents 0.5 km centered at alt? Or starting at alt?
                # GloSSAC docs usually say 0.5 km grid. 
                # If we sum ext * 0.5, we assume ext is constant over the 0.5 km bin.
                lat_saod_comp.append(np.sum(curr_ext * dz))
            else:
                lat_saod_comp.append(np.nan)
                
            # File SAOD
            curr_od = od_file_trop[t_idx, l_idx]
            if not np.ma.is_masked(curr_od):
                lat_saod_file.append(curr_od)
            else:
                lat_saod_file.append(np.nan)
        
        # Weighted average over latitudes
        lat_saod_comp = np.array(lat_saod_comp)
        valid_comp = ~np.isnan(lat_saod_comp)
        if np.any(valid_comp):
            saod_comp[t_idx] = np.sum(lat_saod_comp[valid_comp] * weights[valid_comp]) / np.sum(weights[valid_comp])
        else:
            saod_comp[t_idx] = np.nan

        lat_saod_file = np.array(lat_saod_file)
        valid_file = ~np.isnan(lat_saod_file)
        if np.any(valid_file):
            saod_file[t_idx] = np.sum(lat_saod_file[valid_file] * weights[valid_file]) / np.sum(weights[valid_file])
        else:
            saod_file[t_idx] = np.nan

    return time, saod_comp, saod_file
