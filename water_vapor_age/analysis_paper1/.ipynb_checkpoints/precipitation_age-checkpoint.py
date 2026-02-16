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

def age_precip(tau3d, dq3d,pfull,ps):
    time,plev,lon, lat = tau3d.shape
    mu_2d = np.zeros((time,lon, lat))
    for t in range(time):
        for x in range(lon):
            for y in range(lat):
                tau = tau3d[t,:,x, y,]
                dq = dq3d[t,:,x, y]

                prev_age = 0
                cur_mu = 0
                prev_dq = dq[0]
                dq_l = prev_dq

                for i in range(1, plev):
                    cur_age = tau[i]
                    cur_dq = dq[i]
                    cur_dq_l = cur_dq + dq_l

                    if cur_dq < 0:  # precipitation
                        cur_mu = (cur_age * cur_dq + dq_l * cur_mu) / cur_dq_l

                    # dq_l gets updated regardless
                    dq_l = 0 if cur_dq_l > 0 else cur_dq_l

                mu_2d[t,x, y] = cur_mu

    return mu_2d



def get_age_precip(ds):

    # Use Dask-aware arrays

    qT = ds.sphum_age_1
    q = ds.sphum
    tau = qT / q
    ps = ds.ps.values/100
    pfull = ds.pfull.values
    dq_conv = ds.dt_qg_convection
    dq_cond = ds.dt_qg_condensation
    P_cond = ds.condensation_rain  # already in kg/kg/s
    P_conv = ds.convection_rain

    # Efficient parallel apply using Dask
    p_age_cond =age_precip(tau.values,dq_cond.values,pfull,ps)

    p_age_conv =age_precip(tau.values,dq_conv.values,pfull,ps)
    # Avoid divide-by-zero
    total_precip = P_conv + P_cond
    p_age = xr.where(total_precip > 0,
                     (p_age_cond * P_cond + p_age_conv * P_conv) / total_precip,
                     0.0)

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
    save_ds(ds_precip,exp_save_name,"precipitation_age_mean")
    

if __name__ == "__main__":
    exp_folder_name = sys.argv[1]
    print(f"starting the script {exp_folder_name}",file=sys.stdout,flush=True)
    process_precip_age(exp_folder_name)
    

