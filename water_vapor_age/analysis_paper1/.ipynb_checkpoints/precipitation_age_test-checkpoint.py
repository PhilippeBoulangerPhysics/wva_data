import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import dask
from dask.diagnostics import ProgressBar
os.chdir("/home/philbou/projects/def-rfajber/philbou/analysis_paper1")
import sys
import diagnostic_plot_helper as dps  
from scipy.special import gamma
import gc

import cartopy.crs as ccrs

import numpy as np
import xarray as xr

def age_precip(tau3d, dq3d, pfull, ps):
    time, plev, lon, lat = tau3d.shape
    mu_2d = np.zeros((time, lon, lat))

    for t in range(time):
        for x in range(lon):
            for y in range(lat):
                tau_col = tau3d[t, :, x, y]
                dq_col = dq3d[t, :, x, y]

                # Ensure top of atmosphere -> surface order
                if pfull[0] < pfull[-1]:
                    tau_col = tau_col[::-1]
                    dq_col = dq_col[::-1]

                dq_l_prev = 0.0
                mu_prev = 0.0

                for i in range(len(pfull)):
                    dq_i = dq_col[i]
                    tau_i = tau_col[i]

                    dq_l = max(-(np.sum(dq_col[:i+1])), 0)  # Eq. (dql)

                    if dq_i < 0:  # condensation adds new liquid
                        dq_i_abs = -dq_i
                        dq_l_prev_abs = dq_l_prev
                        if dq_l > 0:
                            mu_i = (dq_i_abs * tau_i + dq_l_prev_abs * mu_prev) / dq_l
                        else:
                            mu_i = mu_prev
                    else:  # evaporation
                        mu_i = mu_prev

                    dq_l_prev = dq_l
                    mu_prev = mu_i

                # Mean age of precip at surface
                mu_2d[t, x, y] = mu_prev

    return mu_2d




def get_age_precip(ds):
    qT = ds.sphum_age_1
    q = ds.sphum
    tau = qT / q
    dq_conv = ds.dt_qg_convection
    dq_cond = ds.dt_qg_condensation
    P_cond = ds.condensation_rain
    P_conv = ds.convection_rain

    pfull = ds.pfull.values
    ps = ds.ps.values / 100

    p_age_cond = age_precip(tau.values, dq_cond.values, pfull, ps)
    p_age_conv = age_precip(tau.values, dq_conv.values, pfull, ps)

    total_precip = P_conv + P_cond
    p_age = xr.where(
        total_precip > 0,
        (p_age_cond * P_cond + p_age_conv * P_conv) / total_precip,
        0.0
    )

    return p_age



def monthly_avg(ds):
    # Use Dask-aware resampling
    n_steps = ds.dims["time"]
    steps_per_month = 120
    n_months = n_steps // steps_per_month

    # Trim to full months only
    ds_trimmed = ds.isel(time=slice(0, n_months * steps_per_month))
    ds_trimmed = ds_trimmed.chunk({'time': steps_per_month})  # Match coarsen window
    monthly_ds = ds_trimmed.coarsen(time=steps_per_month, boundary="trim").mean()
    return monthly_ds

def save_ds(ds,exp_save_name,ds_name,avg = False):
    default_dir = "/home/philbou/projects/def-rfajber/philbou/saved_ds"
    folder_path = os.path.join(default_dir, exp_save_name)
    file_path = os.path.join(folder_path, f"{ds_name}.nc")
    if avg:
        ds_avg = monthly_avg(ds)
    else: 
        ds_avg = ds
        
    ds_avg.to_netcdf(
        file_path,
        mode="w",
        format="NETCDF4",
        compute=True  # Forces execution, good for Dask-backed datasets
    )

    print(f"Saved: {file_path}")
    
    
def process_precip_age(exp_save_name):
    
    print("Step 1: Load dataset for precip age",file=sys.stdout, flush=True)
    default_dir = "/home/philbou/projects/def-rfajber/philbou/saved_ds"
    folder_path = os.path.join(default_dir, exp_save_name)
    file_path = os.path.join(folder_path, f"age_precip.nc")
    ds = xr.open_mfdataset(file_path)  # must return Dask-backed Dataset
    print("Dataset loaded",file=sys.stdout, flush=True)
    precip_age = get_age_precip(ds)
    ds_precip = precip_age.to_dataset(name="precip_age")
    print("saving age.nc",file=sys.stdout,flush=True)
    save_ds(ds_precip,exp_save_name,"precipitation_age_test")
    

if __name__ == "__main__":
    exp_folder_name = sys.argv[1]
    print(f"starting the script {exp_folder_name}",file=sys.stdout,flush=True)
    process_precip_age(exp_folder_name)
    

