import xarray as xr
import os
import numpy as np

class Diagnostic:
    def __init__(self, name: str, data: xr.DataArray):
        self.name = name
        self.data = data
        self.time = None
        self.zonal = None
        self.time_zonal = None
        dims = self.data.dims
        if "time" in dims:
            self.time = self.data.mean("time",skipna=True)
        if "lon" in dims:
            self.zonal = self.data.mean("lon",skipna=True)
        if "time" in dims and "lon" in dims:
            self.time_zonal = self.data.mean(["time", "lon"],skipna=True)
    
    def update_averges(self):
        dims = self.data.dims
        if "time" in dims:
            self.time = self.data.mean("time",skipna=True)
        if "lon" in dims:
            self.zonal = self.data.mean("lon",skipna=True)
        if "time" in dims and "lon" in dims:
            self.time_zonal = self.data.mean(["time", "lon"],skipna=True)
    
    def get_global_mean(self):
        print("To be implemented: global mean calculation for diagnostic:", self.name)
        return None

    
class ExperimentLoader:
    def __init__(self, experiment_name: str, data_dir: str = "/home/philbou/projects/def-rfajber/philbou/wva_data/data/split_datasets"):
        self.experiment_name = experiment_name
        self.data_dir = data_dir
        self.experiment_folder = os.path.join(self.data_dir, self.experiment_name)
        self._height_mask = None
        
    def load_diagnostic(self, diagnostic_name: str, is_monthly: bool = False,coord = "") -> xr.Dataset:
        """ Load a specific diagnostic dataset for the experiment. """
        frequency = "monthly/" if is_monthly else "daily/"
        diagnostic_file = os.path.join(self.experiment_folder, f"{frequency}{diagnostic_name}.nc")
        if not os.path.exists(diagnostic_file):
            raise FileNotFoundError(f"Diagnostic file {diagnostic_file} not found for experiment {self.experiment_name}.")
        
        diagnostic_name_ext = f"{coord}{diagnostic_name}_{frequency[:-1]}"  # Append frequency to diagnostic name for attribute naming
        diagnostic = Diagnostic(diagnostic_name_ext, xr.open_mfdataset(diagnostic_file)[diagnostic_name])
        self.__setattr__(diagnostic_name_ext, diagnostic)

    def apply_height_mask(self):
        if self._height_mask is None:
            raise ValueError("Height mask not loaded. Please load the height mask diagnostic first.")
        mask = self._height_mask.data
        for attr_name in dir(self):
            if attr_name.startswith("_") is not True:
                obj = getattr(self, attr_name)
                if isinstance(obj, Diagnostic):
                    obj.data = obj.data.where(~mask)
                    obj.update_averges()