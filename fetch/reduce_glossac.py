"""Write the GloSSAC V2.23 subset shipped with this repository.

The paper uses GloSSAC only at 525 nm (and 1020 nm in one legacy plot), at all
latitudes (the tropical mean, the 20-25 N bin and the zonal field of Sect. 6),
from the tropopause to 40 km.
This script cuts exactly that slice out of the full 530-MB NASA file
(https://asdc.larc.nasa.gov/project/GloSSAC), keeping the variable names and
conventions the package's loader expects, so every script runs unchanged on
the subset.  Run it only if you want to regenerate data/glossac/ from the
original file:

    python fetch/reduce_glossac.py /path/to/GloSSAC_V2.23_NC4.nc

GloSSAC is a NASA data product (Thomason et al., 2018, doi:10.5194/essd-10-469-2018);
the subset is a derived extract and carries no additional rights.
"""
import sys
from pathlib import Path
import numpy as np
import netCDF4 as nc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "glossac" / "GloSSAC_V2.23_subset.nc"
LAT_MIN, LAT_MAX = -90.0, 90.0          # all latitude bins (Sect. 6 uses the zonal 525-nm field)
WAVELENGTHS_NM = (525.0, 1020.0)
ALT_MAX_KM = 40.5


def main(src):
    ds = nc.Dataset(src)
    lat = ds["lat"][:]; alt = ds["alt"][:]; wls = ds["wavelengths_glossac"][:]
    li = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    ai = np.where(alt <= ALT_MAX_KM)[0]
    wi = np.array([int(np.argmin(np.abs(wls - w))) for w in WAVELENGTHS_NM])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    o = nc.Dataset(OUT, "w", format="NETCDF4")
    o.setncattr("title", "GloSSAC V2.23 subset (525 and 1020 nm; all latitudes; <=40 km) for stardust-sai-monitor")
    o.setncattr("source", str(Path(src).name))
    o.setncattr("history", "fetch/reduce_glossac.py")
    for name, n in [("time", len(ds["time"][:])), ("lat", len(li)), ("alt", len(ai)),
                    ("wavelengths_glossac", len(wi)), ("month", 12)]:
        o.createDimension(name, n)
    def copy(name, dims, data, **kw):
        v = ds[name]
        vo = o.createVariable(name, v.dtype, dims, zlib=True, complevel=6, **kw)
        for a in v.ncattrs():
            if a != "_FillValue": vo.setncattr(a, v.getncattr(a))
        vo[:] = data
    copy("time", ("time",), ds["time"][:])
    copy("lat", ("lat",), lat[li])
    copy("alt", ("alt",), alt[ai])
    copy("wavelengths_glossac", ("wavelengths_glossac",), wls[wi])
    trp = ds["trp_hgt"][:]
    copy("trp_hgt", ds["trp_hgt"].dimensions, trp[li, :] if trp.shape[0] == len(lat) else trp[:, li].T)
    ext = ds["Glossac_Aerosol_Extinction_Coefficient"]
    copy("Glossac_Aerosol_Extinction_Coefficient", ext.dimensions,
         ext[:, li, :, :][:, :, ai, :][:, :, :, wi], fill_value=-999.0)
    od = ds["Glossac_Aerosol_Optical_Depth"]
    copy("Glossac_Aerosol_Optical_Depth", od.dimensions, od[:, li, :][:, :, wi], fill_value=-999.0)
    o.close(); ds.close()
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB): lat {lat[li].min()}..{lat[li].max()}, "
          f"{len(ai)} altitudes, wavelengths {wls[wi]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "Python" / "GloSSAC" / "GloSSAC_V2.23_NC4.nc"))
