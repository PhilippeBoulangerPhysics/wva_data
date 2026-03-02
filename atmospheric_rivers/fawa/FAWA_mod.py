
import diagnostic_plot_helper as dps
import xarray as xr
import numpy as np
import os
import warnings
import matplotlib
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

R = 6371e3
lat_bounds = (0, 90)
lon_bounds = (0,360)#(-175 + 360, -30 + 360)
depth_bounds = (0, 1000)
    
def compute_A_timestep(t,W,W_base_interp_np,lon,lat_interp,dlon_tmp,dlat_tmp):
    A = np.zeros((len(lat_interp),len(lon)))
    D = np.zeros_like(A)
    CWV_ = W[t,:,:]
    for lat_i in range(len(lat_interp)):
        for lon_i in range(len(lon)):
            A_minus = 0
            A_plus = 0
            phi_contour = lat_interp[lat_i]
            m_i = CWV_[lat_i,lon_i]
            idx = np.abs(W_base_interp_np - m_i).argmin()
            cur_phi = lat_interp[idx]
            cur_delta = phi_contour - cur_phi
            D[lat_i,lon_i] = cur_delta
            bot, top = min(cur_phi, phi_contour), max(cur_phi, phi_contour)
            bot_id = np.argmin(np.abs(lat_interp - bot))
            top_id = np.argmin(np.abs(lat_interp - top))
            col_sum = np.sum(CWV_[bot_id:top_id, lon_i] * R**2 * np.cos(np.deg2rad(lat_interp[bot_id:top_id])) * dlon_tmp * dlat_tmp)
            if cur_delta > 0:
                A_plus += col_sum
            elif cur_delta < 0:
                A_minus += col_sum
            
            A[lat_i,lon_i]+= (A_plus - A_minus) / (2*np.pi * R * np.cos(np.deg2rad(phi_contour)))
    return A


class FAWA:
    def __init__(self, exp_folder_name, month_id):
        self.base_dir = "/home/philbou/projects/def-rfajber/philbou/wva_data/data/split_datasets/"
        self.exp_folder_name = exp_folder_name
        self.exp_data_path = os.path.join(self.base_dir, self.exp_folder_name)
        self.month_id = month_id


def select_region3D(ds, lat_range, lon_range, pfull_range,ph = True):
    if ph :
        region = ds.sel(
            lat=slice(lat_range[0], lat_range[1]),
            lon=slice(lon_range[0], lon_range[1]),
            pfull=slice(pfull_range[0], pfull_range[1]),    
            phalf=slice(pfull_range[0], pfull_range[1])
        )
    else:
        region = ds.sel(
            lat=slice(lat_range[0], lat_range[1]),
            lon=slice(lon_range[0], lon_range[1]),
            pfull=slice(pfull_range[0], pfull_range[1])        )
    return region
def select_region2D(ds, lat_range, lon_range):
    region = ds.sel(
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1])
    )
    return region

def save_moist_diag_separatly(exp_folder_name, month_id, file_name="atmos_monthly.nc"):
    """ 
    Saves moiture related diagnostics for the finite wave amplitude calculations.
    """
    base_dir = os.environ["GFDL_DATA"]
    in_path = f"{base_dir}/{exp_folder_name}/run{month_id:04d}/{file_name}"
    in_path_2 = f"{base_dir}/{exp_folder_name}/run{month_id:04d}/page_shape_{file_name}"
    ds_ = xr.open_dataset(in_path)
    ds_2_ = xr.open_dataset(in_path_2)
    ds = select_region3D(ds_, lat_bounds, lon_bounds, depth_bounds)
    shape = select_region3D(ds_2_.shape,lat_bounds, lon_bounds,depth_bounds,ph = False)
    p_age = select_region2D(ds_2_.precip_age,lat_bounds, lon_bounds)
    P = 86400 * ds.precipitation
    E = 86400 * ds.flux_lhe /2.5e6
    q = ds.sphum
    qT = ds.sphum_age_1
    T = (qT / q)/(24*(60**2))
    precip_age = p_age.where(p_age != 0)

    q_np = q.values
    ps_np = ds.ps.mean(dim= "time").values
    phalf_np = 100*ds.phalf.values
    bk_np = ds.bk.values
    T_np = T.values
    W_np = dps.vertical_int(q_np,phalf_np,ps_np)
    T_V_np = dps.vertical_int_moist(T_np,q_np,phalf_np,ps_np)
    W_daily = xr.DataArray(
    W_np,
    dims=("time", "lat", "lon"),
    coords={
        "time": ds.time,
        "lat": ds.lat,
        "lon": ds.lon
    },
    name="W"
    )
    W_base = W_daily.mean("lon")
    eqlat = ds.lat.values.copy()
    lat_interp = np.linspace(eqlat[0],eqlat[-1],100)
    W_base_interp = W_base.mean("time").interp(lat =lat_interp).values

    W_daily_interp = W_daily.interp(lat =lat_interp).values
    
    A_time_lat_lon = np.zeros_like(W_daily_interp)

    iterator = range(len(ds.time))
    dlon_tmp = (ds.lon.values[1] - ds.lon.values[0])
    dlat_tmp = lat_interp[1] - lat_interp[0]
    
    for t in iterator:
        A_time_lat_lon[t, :, :]= compute_A_timestep(t,W_daily_interp,W_base_interp,ds.lon.values,lat_interp,dlon_tmp,dlat_tmp)
    ds_out = xr.Dataset(
        {
            "P": P.assign_attrs(units="mm/day"),
            "E": E.assign_attrs(units="mm/day"),

            "q": (1e3 * q).assign_attrs(units="g/kg"),
            "qT": qT.assign_attrs(units="days * g/kg"),

            "T": T.assign_attrs(units="model_days"),
            "shape": shape.assign_attrs(units=""),

            "precip_age": (precip_age / (24 * 60**2)).assign_attrs(units="model_days"),

            "W": xr.DataArray(
                1e3 * W_np,
                dims=("time", "lat", "lon"),
                coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
                attrs={"units": "g/kg"},
            ),

            "zonal_W": xr.DataArray(
                1e3 * W_base.values,
                dims=("time", "lat"),
                coords={"time": ds.time, "lat": ds.lat},
                attrs={"units": "g/kg"},
            ),

            "T_V": xr.DataArray(
                T_V_np,
                dims=("time", "lat", "lon"),
                coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
                attrs={"units": "model_days"},
            ),

            "CWV_A": xr.DataArray(
                A_time_lat_lon,
                dims=("time", "lat_interp", "lon"),
                coords={"time": ds.time, "lat_interp": lat_interp, "lon": ds.lon},
                attrs={"units": "kg m"},
            ),
        },
        coords={
            "time": ds.time,
            "lat": ds.lat,
            "lat_interp": lat_interp,
            "lon": ds.lon,
            "pfull": ds.pfull,
        },
    )

    ds_12h_sum = 0.25 * ds_out.resample(time="12H").sum()
    ds_12h_mean = ds_out.resample(time="12H").mean(skipna=True)

    ds_12h = xr.Dataset({
        "P": ds_12h_sum.P.assign_attrs(units="mm"),
        "E": ds_12h_sum.E.assign_attrs(units="mm"),
        "q": ds_12h_mean.q,
        "qT": ds_12h_mean.qT,
        "T": ds_12h_mean.T,
        "shape": ds_12h_mean.shape,
        "precip_age": ds_12h_mean.precip_age,
        "W": ds_12h_mean.W,
        "zonal_W": ds_12h_mean.zonal_W,
        "T_V": ds_12h_mean.T_V,
        "CWV_A": ds_12h_mean.CWV_A,
    })

    out_path = f"{base_dir}/{exp_folder_name}/run{month_id:04d}/moist_data.nc"
    # --- 6. Save to NetCDF ---
    if os.path.exists(out_path):
        os.remove(out_path)

    ds_out.to_netcdf(out_path, mode="w")
    
    out_path_12h = f"{base_dir}/{exp_folder_name}/run{month_id:04d}/moist_data_12h.nc"
    # --- 6. Save to NetCDF ---
    if os.path.exists(out_path_12h):
        os.remove(out_path_12h)

    ds_12h.to_netcdf(out_path_12h, mode="w")
