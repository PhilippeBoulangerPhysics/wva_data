import numpy as np
import xarray as xr
import os
import dask
from dask.diagnostics import ProgressBar
os.chdir("/home/philbou/projects/def-rfajber/philbou/analysis_paper1")
import sys
import diagnostic_plot_helper as dps  
from scipy.special import gamma
import gc

def split_exp(ds, exp_save_name,list_names = ["age","precip_age","dynamics","mixed_layer","atmosphere","rrtm_rad","two_stream"]):
    diag_groups = {
    
        "age": ['precip_age','sphum_age_1','sphum_age_2','sphum_age_3','sphum_age_4','sphum_age_5','sphum_age_6','sphum','precipitation','dt_sink',
                'height','phalf','ps','latb','lonb','dt_qg_convection','dt_qg_condensation','dt_qg_diffusion','condensation_rain','convection_rain'],
        "precip_age": ['precip_age','sphum_age_1','sphum_age_2','sphum_age_3','sphum_age_4','sphum','precipitation','dt_sink',
                'height','phalf','ps','latb','lonb','dt_qg_convection','dt_qg_condensation','dt_qg_diffusion','condensation_rain','convection_rain','ucomp','vcomp','omega'],
        "dynamics": ['ps','bk','pk','sphum','ucomp','vcomp','omega','height','temp','vor','div'],
        "mixed_layer" : ['t_surf','flux_lhe', 'flux_t', 'flux_oceanq', 'corr_flux'],
        
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
    
    

def process_run(exp_folder_name,exp_save_name,start,end,file_name="atmos_monthly.nc",
                list_names_to_do = ["age","precip_age","dynamics","mixed_layer","atmosphere","rrtm_rad","two_stream"]):
    print("Step 1: Loading entiredataset with Dask")
    ds = dps.open_experiment(exp_folder_name, start, end, file_name)  # must return Dask-backed Dataset
    split_ds_dict = split_exp(ds, exp_save_name,list_names = list_names_to_do)
    print(f"Dataset loaded: {split_ds_dict.keys()}")
    for key in split_ds_dict.keys():
        if key != "age_precip":
            save_ds(split_ds_dict[key],exp_save_name,key,avg = True)
        else: 
            save_ds(split_ds_dict[key],exp_save_name,key,avg = False)
    ds.close()
    del ds
    print("exp split")
    
if __name__ == "__main__":
    exp_folder_name = sys.argv[1]
    print(f"starting the script {exp_folder_name}")
    process_run(exp_folder_name,f"{exp_folder_name}_",347,358,file_name="atmos_monthly.nc",
                list_names_to_do = ["precip_age"])#["age","precip_age","dynamics","mixed_layer","atmosphere","rrtm_rad","two_stream"])