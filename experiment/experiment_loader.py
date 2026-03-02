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
    
    def get_global_mean(self):
        print("To be implemented: global mean calculation for diagnostic:", self.name)
        return None
    
class ExperimentLoader:
    def __init__(self, experiment_name: str, data_dir: str = "/home/philbou/projects/def-rfajber/philbou/wva_data/data/split_datasets"):
        self.experiment_name = experiment_name
        self.data_dir = data_dir
        self.experiment_folder = os.path.join(self.data_dir, self.experiment_name)
    
    def load_diagnostic(self, diagnostic_name: str, is_monthly: bool = False) -> xr.Dataset:
        """ Load a specific diagnostic dataset for the experiment. """
        diagnostic_file = os.path.join(self.experiment_folder, f"{diagnostic_name}{'_monthly' if is_monthly else ''}.nc")
        if not os.path.exists(diagnostic_file):
            raise FileNotFoundError(f"Diagnostic file {diagnostic_file} not found for experiment {self.experiment_name}.")
        
        diagnostic = Diagnostic(diagnostic_name, xr.open_dataset(diagnostic_file))
        dims = diagnostic.data.dims
        if "time" in dims:
            diagnostic.time = diagnostic.data.mean("time")
        if "lon" in dims:
            diagnostic.zonal = diagnostic.data.mean("lon")
        if "time" in dims and "lon" in dims:
            diagnostic.time_zonal = diagnostic.data.mean(["time", "lon"])
        
        return diagnostic