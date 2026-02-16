import sys
from composite import get_composite_data

month_id = int(sys.argv[1])

print(f"Processing month {month_id}",file=sys.stdout, flush=True)
get_composite_data("RT42_sst_0_bucket", month_id)
print(f"Saved month {month_id}",file=sys.stdout, flush=True)
