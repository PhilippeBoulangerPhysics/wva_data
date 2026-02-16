import os
from process_run import add_page_shape_to_diag

# SLURM info
proc_id = int(os.environ["SLURM_PROCID"])   # 0..59
n_tasks = int(os.environ["SLURM_NTASKS"])   # 60

START = 60
END = 96   # inclusive

# Loop over jobs assigned to this task
for month_id in range(START + proc_id, END + 1, n_tasks):
    print(f"[Task {proc_id}] Running month_id = {month_id}")

    add_page_shape_to_diag(
        "RT85_sst_0",
        month_id,
        file_name="atmos_monthly.nc"
    )

    print(f"[Task {proc_id}] Saved month_id = {month_id}")