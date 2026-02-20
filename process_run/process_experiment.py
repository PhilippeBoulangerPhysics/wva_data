import model_data_loader as mdl
import sys
from datetime import datetime as dt

    
DIAGNOSTIC_VARIABLES = [
    "ps",
    "bk",
    "pk",
    "precipitation",
    "t_surf",
    "flux_lhe",
    "flux_t",
    "flux_oceanq",
    "corr_flux",
    "albedo",
    "sphum",
    "ucomp",
    "vcomp",
    "omega",
    "wspd",
    "height",
    "temp",
    "vor",
    "div",
    "sphum_age_1",
    "sphum_age_2",
    "cape",
    "dt_qg_convection",
    "dt_qg_condensation",
    "dt_sink",
    "dt_tracer",
    "dt_tracer_diff",
    "dt_qg_diffusion",
    "rh",
    "condensation_rain",
    "convection_rain",
    "olr",
    "toa_sw",
    "tdt_rad",
    "tdt_sw",
    "tdt_lw",
    "flux_sw",
    "flux_lw",
    "precipitation_age",
    "mean_age",
    "shape_parameter"
]

def add_age_diagnostics_to_monthly_dataset(experiment_name : str, month : int):
    """ Load the dataset for a specific experiment and month, calculate the precipitation age, and save the modified dataset."""
    t0_ = dt.now()
    monthly_dataset = mdl.MonthlyDataset(experiment_name, month)
    monthly_dataset.add_precipitation_age_to_ds()
    monthly_dataset.save_dataset()
    t_final = dt.now()
    logger.log(f"Saved age diagnostics for month {month} of experiment: {experiment_name}: {t_final-t0_}")


class Logger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def log(self, message):
        print(f"[{dt.now()}] {message}", file=sys.stdout, flush=True)


if __name__ == "__main__":
    experiment_name = str(sys.argv[1])
    month_start = int(sys.argv[2])
    month_end = int(sys.argv[3])
    global logger
    logger = Logger(experiment_name)
    
    logger.log(f"Processing experiment: {experiment_name} for months {month_start} to {month_end}")
    for month in range(month_start, month_end + 1):
        add_age_diagnostics_to_monthly_dataset(experiment_name, month)
    
    t0 = dt.now()
    data_obj = mdl.MultiYearDataset(experiment_name, month_start, month_end)
    data_obj.create_saved_data_dir()
    for diag in DIAGNOSTIC_VARIABLES:
        try:
            data_obj.save_diagnostic_dataset(diag)
        except Exception as e:
            logger.log(f"Error saving diagnostic {diag}: {e}")
    t1 = dt.now()
    data_obj.save_namelist()
    logger.log(f"Saved all diagnostic datasets for experiment: {experiment_name} in {t1-t0}")