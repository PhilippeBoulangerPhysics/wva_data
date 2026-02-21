from cmath import tau
import os
import f90nml
import numpy as np
import xarray as xr

SCRATCH_DATA_PATH = "/home/philbou/scratch/isca_data/"
PROJECT_DIR = "/home/philbou/projects/def-rfajber/philbou/wva_data/"


class MonthlyDataset:
    def __init__(self, experiment_name : str, month : int):
        """ Create a MonthlyDataset instance for a specific experiment and month."""
        self.experiment_name = experiment_name
        self.month = month
        self.ds = self.load_dataset()

    def load_dataset(self):
        """ Load the dataset for the given experiment and month."""
        self.input_file_path = f"{SCRATCH_DATA_PATH}{self.experiment_name}/run{self.month:04d}/atmos_monthly"
        return xr.open_mfdataset(f"{self.input_file_path}.nc")

    def _set_variables_precipitation_age(self):
        """ Sets the variables needed for the precipitation age calculation."""
        qT = self.ds.sphum_age_1
        self.q = self.ds.sphum
        safe_q = xr.where(self.q == 0, np.nan, self.q)
        self.tau = qT / safe_q
        self.shape = self.ds.sphum_age_2 / safe_q
        self.ps = self.ds.ps.values/100
        self.pfull = self.ds.pfull.values
        self.dq_conv = self.ds.dt_qg_convection
        self.dq_cond = self.ds.dt_qg_condensation
        self.P_cond = self.ds.condensation_rain  # already in kg/kg/s
        self.P_conv = self.ds.convection_rain

    def _get_precipitation_age_general(self, tau : xr.DataArray, dq : xr.DataArray):
        """ General method to calculate the precipitation age for either convection or condensation."""
        time, pfull, lon, lat = tau.shape
        cur_mu = 0
        prev_dq = dq.isel(pfull=0)
        dq_l = prev_dq
        for i in range(1, pfull):
            cur_age = tau.isel(pfull=i)
            cur_dq = dq.isel(pfull=i)
            cur_dq_l = cur_dq + dq_l

            cur_mu = xr.where(cur_dq < 0,  # condition
                  (cur_age * cur_dq + dq_l * cur_mu) / cur_dq_l,  # value if True
                  cur_mu) 

            dq_l = xr.where(cur_dq_l > 0, 0, cur_dq_l)
        return cur_mu
    
    def get_precipitation_age(self):
        """ Calculate the precipitation age using the general method for both convection and condensation. """
        precipitation_age_cond = self._get_precipitation_age_general(self.tau, self.dq_cond)
        precipitation_age_conv = self._get_precipitation_age_general(self.tau, self.dq_conv)
        total_precip = self.P_conv + self.P_cond
        precipitation_age = xr.where(total_precip > 0,
                        (precipitation_age_cond * self.P_cond + precipitation_age_conv * self.P_conv) / total_precip,
                        0.0)
        return precipitation_age
    
    
    def add_precipitation_age_to_ds(self):
        """ Add the calculated precipitation age to the dataset."""
        self._set_variables_precipitation_age()
        precipitation_age = self.get_precipitation_age()
        self.ds["precipitation_age"] = precipitation_age
        self.ds.precipitation_age.attrs["units"] = "s"
        self.ds.precipitation_age.attrs["long_name"] = "Mean Age of Precipitation"
        self.ds["mean_age"] = self.tau
        self.ds.mean_age.attrs["units"] = "s"
        self.ds.mean_age.attrs["long_name"] = "Mean Age of Water Vapor"
        self.ds["shape_parameter"] = self.shape
        self.ds.shape_parameter.attrs["units"] = "none"
        self.ds.shape_parameter.attrs["long_name"] = "Shape Parameter of Water Vapor Age Distribution"

    def save_dataset(self):
        """Save the modified dataset with the added precipitation age variables to a new NetCDF file with a '_with_precipitation_age' suffix."""
        self.ds.to_netcdf(f"{self.input_file_path}_with_precipitation_age.nc", mode = "w")
        
class MultiYearDataset(MonthlyDataset):
    def __init__(self, experiment_name : str, month_start : int, month_end : int):
        """ Create a MultiYear instance for a specific experiment over multiple months."""
        self.experiment_name = experiment_name
        self.month_start = month_start
        self.month_end = month_end
        self.input_dir = f"{SCRATCH_DATA_PATH}{self.experiment_name}/"
        self.ds = self.load_dataset()

    def create_saved_data_dir(self):
        self.output_dir = f"{PROJECT_DIR}/data/split_datasets/{self.experiment_name}/"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_namelist(self):
        namelist_path =  f"{self.input_dir}run{self.month_start:04d}/input.nml"
        namelist = f90nml.read(namelist_path)
        return namelist
    
    def save_namelist(self):
        namelist = self.get_namelist()
        output_namelist_path = f"{self.output_dir}input_namelist.nml"
        f90nml.write(namelist, output_namelist_path,force=True)
    
    def load_dataset(self):
        """ Load the diagnostic dataset for the given experiment and month."""
        file_paths = []
        for month in range(self.month_start, self.month_end + 1):
            file_path = f"{self.input_dir}run{month:04d}/atmos_monthly_with_precipitation_age.nc"
            file_paths.append(file_path)
        return xr.open_mfdataset(file_paths, chunks={"time": 10}, parallel=True)
    
    def save_diagnostic_dataset(self,diagnostic_name):
        """ Save the loaded diagnostic to a new NetCDF file."""
        output_file_path = f"{self.output_dir}{diagnostic_name}.nc"
        try:
            diagnostic = self.ds[diagnostic_name]
            diagnostic.to_netcdf(output_file_path)
            return output_file_path
        except Exception as e:
            raise e