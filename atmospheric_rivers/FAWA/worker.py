import sys
from FAWA_mod import save_moist_diag_separatly

# Get month ID from command line argument
month_id = int(sys.argv[1])

print(f"Processing month {month_id}")
save_moist_diag_separatly("RT85_sst_0", month_id, file_name="atmos_monthly.nc")