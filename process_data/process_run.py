import numpy as np
import xarray as xr
import os
import dask
from dask.diagnostics import ProgressBar
import sys
import diagnostic_plot_helper as dps  
from scipy.special import gamma
import gc
from datetime import datetime

    
def age_precip(tau, dq):
    """
    Vertically sums to get the age of precipitation
    """
    # tau, dq: [time, plev, lat, lon]
    _, plev, _, _ = tau.shape

    cur_mu = np.zeros_like(tau[:, 0, :, :])
    dq_l  = dq[:, 0, :, :].copy()

    for i in range(1, plev):
        cur_age = tau[:, i, :, :]
        cur_dq  = dq[:, i, :, :]
        cur_dq_l = cur_dq + dq_l

        # mask where precipitation occurs
        mask = cur_dq < 0.0

        # Only compute updated values for masked points
        updated_mu = (cur_age * cur_dq + dq_l * cur_mu) / cur_dq_l

        # In-place update — NO np.where
        cur_mu[mask] = updated_mu[mask]

        # update dq_l: zero if cur_dq_l > 0 else keep cur_dq_l
        mask_pos = cur_dq_l > 0
        dq_l[mask_pos] = 0.0
        dq_l[~mask_pos] = cur_dq_l[~mask_pos]

    return cur_mu


def stamp(prev_time, msg):
    """
    Gives date and time to keep track of processes
    """
    now = datetime.now()
    if prev_time is None:
        print(f"[{now.strftime('%H:%M:%S')}] {msg}")
    else:
        dt = (now - prev_time).total_seconds()
        print(f"[{now.strftime('%H:%M:%S')}] (+{dt:.3f}s) {msg}")
    return now


def get_age_precip(tau,dq_cond,dq_conv,P_conv,P_cond):
    """
    Calculates age of precip from cond and conv.
    """
    # ----- CONDITIONAL PRECIPITATION -----
    mu_cond = age_precip(tau, dq_cond)

    # ----- CONVECTIVE PRECIPITATION -----
    mu_conv = age_precip(tau, dq_conv)

    # Combine precipitation components
    total_precip = P_conv + P_cond

    # Initialize output array
    p_age = np.zeros_like(total_precip)

    # Mask where total_precip > 0
    mask = total_precip > 0

    # Compute only where mask is True
    p_age[mask] = (mu_cond[mask] * P_cond[mask] + mu_conv[mask] * P_conv[mask]) / total_precip[mask]

    return p_age

def split_exp(ds, exp_save_name,list_names = ["age","precip_age","dynamics","mixed_layer","atmosphere","rrtm_rad","two_stream"]):
    """
    Splits and save monthly averages of diagnostics.
    """
    diag_groups = {
    
        "age": ['precip_age','shape','sphum_age_1','sphum_age_2','sphum_age_3','sphum_age_4','sphum_age_5','sphum_age_6','sphum','precipitation','dt_sink',
                'height','phalf','ps','latb','lonb','dt_qg_convection','dt_qg_condensation','dt_qg_diffusion','condensation_rain','convection_rain','dt_q','dt_tracer_diff','dt_tracer'],
        "data_6h": ['flux_lhe','sphum','precip_age','sphum_age_1','sphum_age_2','sphum_age_3','sphum_age_4','sphum','precipitation','dt_sink','dt_q','dt_tracer_diff','dt_tracer',
                'height','phalf','ps','latb','lonb','dt_qg_convection','dt_qg_condensation','dt_qg_diffusion','condensation_rain','convection_rain','ucomp','vcomp','omega'],
        "dynamics": ['ps','bk','pk','sphum','ucomp','vcomp','omega','height','temp','vor','div'],
        "mixed_layer" : ['t_surf','flux_lhe', 'flux_t', 'flux_oceanq', 'corr_flux','albedo'],
        
        "atmosphere": ['precipitation','cape','rh'],
        "rrtm_rad": ['olr','toa_sw','tdt_rad','tdt_sw_rad','tdt_lw_rad','flux_sw'],
        "two_stream": ['olr','swdn_sfc', 'swdn_toa', 'lwup_sfc', 'lwdn_sfc','net_lw_surf', 'flux_rad', 'flux_lw', 'flux_sw']
    }

    default_dir = "/home/philbou/projects/def-rfajber/philbou/saved_ds"
    folder_path = os.path.join(default_dir, exp_save_name)
    os.makedirs(folder_path, exist_ok=True)

    print(f"Saving datasets to: {folder_path}")
    dict_ds = {}
    for group_name in list_names:
        variable_list = diag_groups[group_name]
        existing_vars = [var for var in variable_list if var in ds.variables]
        if not existing_vars:
            continue  # Skip empty groups

        subset = ds[existing_vars]
        dict_ds[group_name] = subset
    return dict_ds

def monthly_avg(ds):
    """
    Calculates monthly average.
    """
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
    """
    Saves dataset.
    """
    default_dir = "/home/philbou/projects/def-rfajber/philbou/saved_ds"
    folder_path = os.path.join(default_dir, exp_save_name)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{ds_name}.nc")
    if avg:
        ds_avg = monthly_avg(ds)
    else: 
        ds_avg = ds.isel(time=slice(0, None, 24))
        
    ds_avg.to_netcdf(
        file_path,
        mode="w",
        format="NETCDF4",
        compute=True  # Forces execution, good for Dask-backed datasets
    )

    print(f"Saved: {file_path}")
    

def add_page_shape_to_diag(exp_folder_name, month_id, file_name="atmos_monthly.nc"):
    """
    Calculates age of precip and shaope parameter for each timestep and saves into seprate netcdf.
    """
    base_dir = os.environ["GFDL_DATA"]
    out_path = f"{base_dir}/{exp_folder_name}/run{month_id:04d}/{file_name}"
    ds = xr.open_dataset(out_path)
    sphum = ds.sphum.values
    qmoments = [ds.sphum_age_1.values, ds.sphum_age_2.values]

    m1 = qmoments[0] / sphum
    m2 = qmoments[1] / sphum

    var = m2 - m1**2
    std = var ** 0.5

    shape = (m1 / std)
    
    tau = m1
    dq_conv = ds.dt_qg_convection.values
    dq_cond = ds.dt_qg_condensation.values
    P_conv = ds.convection_rain.values
    P_cond = ds.condensation_rain.values
    
    precip_age = get_age_precip(tau,dq_cond,dq_conv,P_conv,P_cond)

    ds_out = xr.Dataset(
        {
            "precip_age": (("time", "lat", "lon"), precip_age),
            "shape": (("time", "pfull", "lat", "lon"), shape),
        },
        coords={
            "time": ds.time,
            "lat": ds.lat,
            "lon": ds.lon,
            "pfull": ds.pfull
        }
    )
    out_path = f"{base_dir}/{exp_folder_name}/run{month_id:04d}/page_shape_{file_name}"
    if os.path.exists(out_path):
        os.remove(out_path)
    ds_out.to_netcdf(out_path, mode="w")
    


def process_run(exp_folder_name,exp_save_name,start,end,file_name="atmos_monthly.nc",
                list_names_to_do = ["age","data_6h","dynamics","mixed_layer","atmosphere","rrtm_rad","two_stream"]
                ,only_precip_age = False):
    print("Step 1: Loading entiredataset with Dask",file=sys.stdout, flush=True)
    ds = dps.open_experiment(exp_folder_name, start, end, file_name)  # must return Dask-backed Dataset
    
    # Deal with precipitation age 
    print("Dataset loaded",file=sys.stdout, flush=True)

    print("saving age.nc",file=sys.stdout,flush=True)
    
    if only_precip_age != True:
        # Split experiment data
        split_ds_dict = split_exp(ds, exp_save_name,list_names = list_names_to_do)
        print(f"Dataset loaded: {split_ds_dict.keys()}",file=sys.stdout, flush=True)
        for key in split_ds_dict.keys():
            if key != "data_6h":
                save_ds(split_ds_dict[key],exp_save_name,key,avg = True)
            else: 
                save_ds(split_ds_dict[key],exp_save_name,key,avg = False)
        ds.close()
        del ds
        print("exp split",file=sys.stdout, flush=True)
    
if __name__ == "__main__":
    exp_folder_name = sys.argv[1]
    print(f"starting the script {exp_folder_name}",file=sys.stdout, flush=True)
    process_run(exp_folder_name,f"{exp_folder_name}",301,360,file_name="atmos_monthly.nc",only_precip_age=False)
    """    num_workers = 8
    print(num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(add_page_shape_to_diag, "RT42_sst_0_bucket",i) for i in  range(301,361,1)]
        for future in as_completed(futures):
            filename = future.result()
            print(f"Saved {filename}")"""