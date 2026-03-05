from cmath import tau
import os
import shutil
from unittest import result
import f90nml
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

SCRATCH_DATA_PATH = "/home/philbou/scratch/isca_data/"
PROJECT_DIR = "/home/philbou/projects/def-rfajber/philbou/wva_data/"
g = 9.81  # Gravitational acceleration in m/s^2
R = 6.371e6  # Earth's radius in meters
SECONDS_TO_DAYS = 1/(24*3600)

class MonthlyDataset:
    def __init__(self, experiment_name : str, month : int, hemisphere : str = None):
        """ Create a MonthlyDataset instance for a specific experiment and month."""
        self.experiment_name = experiment_name
        self.month = month
        self.ds = self.load_dataset()
        if hemisphere:
            self.select_hemisphere(hemisphere)
        self._set_variables()
        self.compute_precipitable_water_vapor()
            
    def select_hemisphere(self,hemisphere):
        """ Select the hemisphere to focus on by slicing the dataset based on latitude. """
        if hemisphere == "NH":
            self.ds = self.ds.sel(lat = slice(0,90))
        elif hemisphere == "SH":
            self.ds = self.ds.sel(lat = slice(-90,0))

    def load_dataset(self):
        """ Load the dataset for the given experiment and month."""
        self.input_file_path = f"{SCRATCH_DATA_PATH}{self.experiment_name}/run{self.month:04d}/atmos_monthly"
        return xr.open_mfdataset(f"{self.input_file_path}.nc",decode_cf=False)

    def _set_variables(self):
        """ Sets the variables needed for the precipitation age calculation."""
        self._set_water_vapor_age()
        self._set_coordinates()
        self._set_water_vapor_tendencies()
        self.compute_precipitable_water_vapor()
        
        
    def _set_water_vapor_age(self):
        """ Set the water vapor variables needed for calculations."""
        qT = self.ds.sphum_age_1
        self.q = self.ds.sphum
        safe_q = xr.where(self.q == 0, np.nan, self.q)
        self.tau = qT / safe_q
        mom2 = self.ds.sphum_age_2/safe_q
        self.standard_deviation = (mom2 - self.tau**2)**0.5
        self.shape = self.standard_deviation / self.tau
    
    def _set_coordinates(self):
        """ Set the coordinate variables needed for calculations."""
        self.ps = self.ds.ps/100
        self.pfull = self.ds.pfull
        self.phalf = self.ds.phalf
        self.lat = self.ds.lat
        self.lon = self.ds.lon
        self.pfull = self.ds.pfull
        self.dp = self.phalf.diff(dim='phalf')
        self.dlon = (self.lon[1] - self.lon[0]).values
        self.dlat = (self.lat[1] - self.lat[0]).values
        self.time = self.ds.time
    
    def _set_water_vapor_tendencies(self):
        """ Set the water vapor tendency variables needed for calculations."""
        self.dq_conv = self.ds.dt_qg_convection
        self.dq_cond = self.ds.dt_qg_condensation
        self.P_cond = self.ds.condensation_rain 
        self.P_conv = self.ds.convection_rain

    def compute_vertically_integrated_age(self):
        tau,q,pfull,ps,dp = self.tau.values,self.q.values,self.pfull.values,self.ps.values,self.dp.values
        self.vertically_integrated_age_np = self.vertical_integral(q*tau,pfull,ps,dp)
        self.vertically_integrated_age = self.create_dataarray_from_np(self.vertically_integrated_age_np, "s", {"time": self.time, "lat": self.lat, "lon": self.lon}, ["time", "lat", "lon"], "Vertically Integrated Mean Age of Water Vapor")
        
    def compute_vertically_integrated_shape(self):
        shape,q,pfull,ps,dp = self.shape.values,self.q.values,self.pfull.values,self.ps.values,self.dp.values
        self.vertically_integrated_shape_np = self.vertical_integral(q*shape,pfull,ps,dp)
        self.vertically_integrated_shape = self.create_dataarray_from_np(self.vertically_integrated_shape_np, "1", {"time": self.time, "lat": self.lat, "lon": self.lon}, ["time", "lat", "lon"], "Vertically Integrated Shape Parameter of Water Vapor Age Distribution")

    def _compute_mu(self, tau : xr.DataArray, dq : xr.DataArray):
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
    
    def compute_precipitation_age(self):
        """ Calculate the precipitation age using the general method for both convection and condensation. """
        precipitation_age_cond = self._compute_mu(self.tau, self.dq_cond)
        precipitation_age_conv = self._compute_mu(self.tau, self.dq_conv)
        total_precip = self.P_conv + self.P_cond
        precipitation_age = xr.where(total_precip > 0,
                        (precipitation_age_cond * self.P_cond + precipitation_age_conv * self.P_conv) / total_precip,
                        0.0)
        self.precipitation_age = precipitation_age
        
    def mask_p_under_ps(self,pfull,ps):
        """ Create a mask for the vertical integral to only include levels where pfull <= ps. """
        mask = pfull[np.newaxis,:,np.newaxis,np.newaxis] <= ps[:,np.newaxis,:,:]
        return mask

    def vertical_integral(self,field,pfull,ps,dp):
        """ Compute the vertical integral of a field using the mask for levels where pfull <= ps."""
        mask = self.mask_p_under_ps(pfull,ps)
        masked_field = mask*field
        tmp = masked_field * dp[np.newaxis,:,np.newaxis,np.newaxis]/g  
        integral = np.sum(tmp,axis = 1)
        return integral
    
    def get_interpolated_baseline_field_and_field(self,field):
        """ Interpolate the baseline field (zonal mean) and the input field to a finer latitude grid. """
        self.set_interpolated_latitude()
        f = interp1d(self.lat, field, axis=1, kind='linear')
        field_interp = f(self.lat_interp)
        field_zonal = np.mean(field_interp,axis = 2)
        field_zonal_time = np.mean(field_zonal,axis = 0)
        return field_zonal_time, field_interp
    
    def set_interpolated_latitude(self):
        """ Set the interpolated latitude grid and create a 2D array of the interpolated latitudes for later use in the activity calculation. """
        self.lat_interp = np.linspace(self.lat[0],self.lat[-1],len(self.lat)*2)
        self.lat_values_2d = np.tile(self.lat_interp[:, np.newaxis], (1, self.lon.size))
    
    def get_equivalent_latitude_for_field(self,baseline,field):
        """ Compute the equivalent latitude for the input field based on the baseline field (zonal mean). """
        diff_base_field =np.abs(baseline[np.newaxis,:,np.newaxis,np.newaxis]-field[np.newaxis,:,:])
        return np.argmin(diff_base_field,axis = 1)[0,:,:]
    
    def compute_activity_along_lat(self,eq,cur,field):
        """ Compute the activity along a latitude circle for the given equivalent latitude and current latitude index. """
        multiplier = np.where(cur - eq >= 0, 1, -1)
        bot = np.minimum(eq, cur)
        top = np.maximum(eq, cur)
        lat_weights = R**2 * np.cos(np.deg2rad(self.lat_interp)) * self.dlat * self.dlon 
        weighted = field * lat_weights[:, None]
        cumsum = np.cumsum(weighted, axis=0)
        cumsum = np.vstack([np.zeros((1, cumsum.shape[1])), cumsum])
        top_p = top + 1
        bot_p = bot
        col_sum = cumsum[top_p, np.arange(len(eq))] - cumsum[bot_p, np.arange(len(eq))]
        A = multiplier * col_sum / (2 * np.pi * R * np.cos(np.deg2rad(self.lat_interp[cur])))
        return A
    
    def compute_2D_activity(self,field,equivalent_latitude):
        """ Compute the 2D activity field by iterating over each latitude index and computing the activity along the latitude circle. """
        activity_2D = []
        for lat_index in range(len(self.lat_interp)):
            lat_index = int(lat_index)
            equivalent_latitude_at_index = equivalent_latitude[lat_index, :]
            diff = np.abs(self.lat_interp[:, None] - equivalent_latitude_at_index[None, :])
            equivalent_latitude_at_index_index = np.argmin(diff, axis=0).astype(int)
            activiti_along_lat = self.compute_activity_along_lat(equivalent_latitude_at_index_index,lat_index,field[lat_index,:])
            activity_2D.append(activiti_along_lat)
        activity_2D = np.array(activity_2D)
        return activity_2D
    
    def compute_activity(self,field_zonal_time,field_interp):
        """ Compute the activity field for the given input field by first interpolating the baseline and input fields, then computing the equivalent latitude and finally computing the 2D activity field. """
        equivalent_latitude = self.get_equivalent_latitude_for_field(field_zonal_time,field_interp)
        activity_2D = self.compute_2D_activity(field_interp,equivalent_latitude)
        return activity_2D
    
    def compute_precipitable_water_vapor(self):
        """ Compute the precipitable water vapor by vertically integrating the specific humidity field. """
        q,pfull,ps,dp = self.q.values,self.pfull.values,self.ps.values,self.dp.values
        column_integrated_water_vapor_np = self.vertical_integral(q,pfull,ps,dp)
        column_integrated_water_vapor_np = column_integrated_water_vapor_np
        self.column_integrated_water_vapor = self.create_dataarray_from_np(column_integrated_water_vapor_np, "kg/m^2", {"time": self.time, "lat": self.lat, "lon": self.lon}, ["time", "lat_interp", "lon"], "Column Integrated Water Vapor") 
        
    def compute_column_integrated_water_vapor_activity(self):
        """ Compute the column integrated activity by vertically integrating the 2D activity field. """
        activity_3D = []
        CWV_zonal_time, CWV_interp = self.get_interpolated_baseline_field_and_field(self.column_integrated_water_vapor)
        for t in range(len(self.time.values)):
            activity_2D = self.compute_activity(CWV_zonal_time, CWV_interp[t])
            activity_3D.append(activity_2D)
            
        self.activity_column_integrated_water_vapor_np = np.array(activity_3D)
        self.activity_column_integrated_water_vapor = self.create_dataarray_from_np(self.activity_column_integrated_water_vapor_np, "kg/m^2/s", {"time": self.time, "lat_interp": self.lat_interp, "lon": self.lon}, ["time", "lat_interp", "lon"], "Activity of Column Integrated Water Vapor")
        
    
    def create_dataarray_from_np(self,data_np,units,coords,dims,long_name):
        """ Create an xarray DataArray for the column integrated water vapor activity field with the appropriate coordinates and attributes. """
        data_array = xr.DataArray(
            data_np,
            coords=coords,
            dims=dims,
            attrs={"units": units, "long_name": long_name}
        )
        return data_array
        
    def add_field_to_ds(self, field_name : str, field_data : xr.DataArray, units : str, long_name : str):
        """ Add a field to the dataset with the specified name, data, units, and long name attributes."""
        self.ds[field_name] = field_data
        self.ds[field_name].attrs["units"] = units
        self.ds[field_name].attrs["long_name"] = long_name
    
    def add_new_fields_to_ds(self):
        """ Add the calculated precipitation age to the dataset."""
        self.compute_precipitation_age()
        self.compute_column_integrated_water_vapor_activity()
        self.compute_vertically_integrated_age()
        self.compute_vertically_integrated_shape()
        
        self.add_field_to_ds("precipitation_age", self.precipitation_age*SECONDS_TO_DAYS, "days", "Mean Age of Precipitation")
        self.add_field_to_ds("column_integrated_water_vapor_activity", self.activity_column_integrated_water_vapor, "kg/m^2/s", "Column Integrated Water Vapor Activity")
        self.add_field_to_ds("column_integrated_water_vapor", self.column_integrated_water_vapor, "kg/m^2", "Column Integrated Water Vapor")

        self.add_field_to_ds("vertically_integrated_mean_age", self.vertically_integrated_age*SECONDS_TO_DAYS, "days", "Vertically Integrated Mean Age of Water Vapor")
        self.add_field_to_ds("vertically_integrated_shape", self.vertically_integrated_shape, "1", "Vertically Integrated Shape Parameter of Water Vapor Age Distribution")
        
        self.add_field_to_ds("mean_age", self.tau*SECONDS_TO_DAYS, "days", "Mean Age of Water Vapor")
        self.add_field_to_ds("shape_parameter", self.shape, "1", "Shape parameter of Water Vapor Age Distribution")
        self.add_field_to_ds("standard_deviation", self.standard_deviation*SECONDS_TO_DAYS, "days", "Standard Deviation of Water Vapor Age Distribution")
    
    def save_dataset(self):
        """ Save the modified dataset with the added precipitation age variables to a new NetCDF file with a '_extra' suffix. """
        self.ds.to_netcdf(f"{self.input_file_path}_wva.nc", mode = "w")
    
    def save_monthly_average(self):
        self.ds.mean(dim="time").to_netcdf(f"{self.input_file_path}_wva_monthly_average.nc", mode = "w")
        
class MultiYearDataset(MonthlyDataset):
    def __init__(self, experiment_name : str, month_start : int, month_end : int):
        """ Create a MultiYear instance for a specific experiment over multiple months."""
        self.experiment_name = experiment_name
        self.month_start = month_start
        self.month_end = month_end
        self.input_dir = f"{SCRATCH_DATA_PATH}{self.experiment_name}/"
        self.ds = self.load_dataset()
        self.ds_monthly = self.load_dataset(is_monthly=True)

    def create_saved_data_dir(self):
        self.output_dir = f"{PROJECT_DIR}/data/split_datasets/{self.experiment_name}/"
        # Remove directory if it exists
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        # Create fresh directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}daily/", exist_ok=True)
        os.makedirs(f"{self.output_dir}monthly/", exist_ok=True)
        
    def get_namelist(self):
        namelist_path =  f"{self.input_dir}run{self.month_start:04d}/input.nml"
        namelist = f90nml.read(namelist_path)
        return namelist
    
    def save_namelist(self):
        namelist = self.get_namelist()
        output_namelist_path = f"{self.output_dir}input_namelist.nml"
        f90nml.write(namelist, output_namelist_path,force=True)
    
    def load_dataset(self,is_monthly=False):
        """ Load the diagnostic dataset for the given experiment and month."""
        file_paths = []
        for month in range(self.month_start, self.month_end + 1):
            file_path = f"{self.input_dir}run{month:04d}/atmos_monthly_wva{'' if not is_monthly else '_monthly_average'}.nc"
            file_paths.append(file_path)
        return xr.open_mfdataset(
            file_paths,
            concat_dim='time',
            combine='nested',
            chunks={"time": 10},
            parallel=True,decode_cf=False
        )
    
    def save_diagnostic(self, diag_name, is_monthly=False):
        """ Save multiple diagnostics efficiently without loading all into memory."""
        encoding_template = {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
        if is_monthly:frequency = "monthly/"
        else: frequency = "daily/"
        
        try:
            diagnostic = self.ds_monthly[diag_name] if is_monthly else self.ds[diag_name]
            output_path = f"{self.output_dir}{frequency}{diag_name}.nc"
            encoding = {diag_name: encoding_template}
            diagnostic.to_netcdf(output_path, encoding=encoding, engine='netcdf4')
            return diag_name, True, output_path
        except Exception as e:
            return  diag_name, False, str(e)
    

        
        
