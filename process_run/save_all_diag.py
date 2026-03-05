import model_data_loader as mdl
import sys
from datetime import datetime as dt

    
DIAGNOSTIC_VARIABLES = [
    "ps",
    "bk",
    "pk",
    "phalf",
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
    "shape_parameter",
    "column_integrated_water_vapor_activity",
    "vertically_integrated_mean_age",
    "vertically_integrated_shape",
    "standard_deviation",
    "column_integrated_water_vapor"
]


class Logger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def log(self, message):
        print(f"[{dt.now()}] {message}", file=sys.stdout, flush=True)


if __name__ == "__main__":
    experiment_name = str(sys.argv[1])
    month_start = int(sys.argv[2])
    month_end = int(sys.argv[3])
    logger = Logger(experiment_name)
    
    t0 = dt.now()
    logger.log(f"Processing experiment: {experiment_name}")
    logger.log(f"Loading dataset from months {month_start} to {month_end}")
    
    data_obj = mdl.MultiYearDataset(experiment_name, month_start, month_end)
    data_obj.create_saved_data_dir()
    logger.log(f"Output directory created: {data_obj.output_dir}")
    
    total_diags = len(DIAGNOSTIC_VARIABLES)
    
    completed = 0
    failed = 0
    
    # Save monthly averages
    logger.log(f"Saving {total_diags} diagnostics (monthly averages)")
    for diag_name in DIAGNOSTIC_VARIABLES:
        result = data_obj.save_diagnostic(diag_name, is_monthly=True)
        if result[1]:
            logger.log(f"✓ Saved {diag_name} (monthly): {result[2]}")
            completed += 1
        else:
            logger.log(f"✗ Failed to save {diag_name} (monthly): {result[2]}")
            failed += 1

    logger.log(f"Monthly average diagnostics complete: {completed} succeeded, {failed} failed")
    
    data_obj.save_namelist()
    logger.log(f"Saved namelist")
    
    completed = 0
    failed = 0
    
    # Save all diagnostics at once
    logger.log(f"Saving {total_diags} diagnostics (regular)")
    for diag_name in DIAGNOSTIC_VARIABLES:
        result = data_obj.save_diagnostic(diag_name, is_monthly=False)
        if result[1]:
            logger.log(f"✓ Saved {diag_name}: {result[2]}")
            completed += 1
        else:
            logger.log(f"✗ Failed to save {diag_name}: {result[2]}")
            failed += 1
    
    logger.log(f"Regular diagnostics complete: {completed} succeeded, {failed} failed")

    
    t1 = dt.now()
    logger.log(f"All diagnostics saved for experiment: {experiment_name} in {t1-t0}")