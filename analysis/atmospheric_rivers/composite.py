import os
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def select_region3D(ds, lat_range, lon_range, pfull_range):
    region = ds.sel(
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1]),
        lat_interp=slice(lat_range[0], lat_range[1]),
        pfull=slice(pfull_range[0], pfull_range[1])    
        #phalf=slice(pfull_range[0], pfull_range[1])
    )
    return region
    
def select_region3D_full(ds, lat_range, lon_range, pfull_range):
    region = ds.sel(
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1]),
        pfull=slice(pfull_range[0], pfull_range[1]),    
        phalf=slice(pfull_range[0], pfull_range[1])
    )
    return region

def select_region2D(ds, lat_range, lon_range):
    region = ds.sel(
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1])
    )
    return region

zones = {
    "NA": {
        "ID": "NA",
        "long_name": "Inland North America",
        "lon_edges": slice(240, 280),
        "lat_edges": slice(30, 60),
        "lon_edges_": slice(-120, -80),
        "lat_edges_": slice(30, 60),
        "center_lon": [260],
        "center_lat": [45],
        "color": "yellow"
    },
    "NATLW": {
        "ID": "NATL",
        "long_name": "North Atlantic",
        "lon_edges": slice(280, 360),
        "lat_edges": slice(30, 60),
        "lon_edges_": slice(-80, 0),
        "lat_edges_": slice(30, 60),
        "center_lon": [312],
        "center_lat": [45],
        "color": "dodgerblue"
    },
    "NATLE": {
        "ID": "NATL",
        "long_name": "North Atlantic",
        "lon_edges": slice(280, 360),
        "lat_edges": slice(30, 60),
        "lon_edges_": slice(-80, 0),
        "lat_edges_": slice(30, 60),
        "center_lon": [345],
        "center_lat": [45],
        "color": "dodgerblue"
    },
    "EU": {
        "ID": "EU",
        "long_name": "Europe",
        "lon_edges": slice(0, 50),
        "lat_edges": slice(30, 60),
        "lon_edges_": slice(0, 50),
        "lat_edges_": slice(30, 60),
        "center_lon": [25],
        "center_lat": [45],
        "color": "blue"
    },
    "CA": {
        "ID": "CA",
        "long_name": "Central Asia",
        "lon_edges": slice(50, 120),
        "lat_edges": slice(30, 60),
        "lon_edges_": slice(50, 120),
        "lat_edges_": slice(30, 60),
        "center_lon": [85],
        "center_lat": [45],
        "color": "red"
    },
    "NPW": {
        "ID": "NPW",
        "long_name": "North Pacific",
        "lon_edges": slice(120, 240),
        "lat_edges": slice(30, 60),
        "lon_edges_": slice(120, -120),
        "lat_edges_": slice(30, 60),
        "center_lon": [150],
        "center_lat": [45],
        "color": "green"
    },
    "NPE": {
        "ID": "NPE",
        "long_name": "North Pacific",
        "lon_edges": slice(120, 240),
        "lat_edges": slice(30, 60),
        "lon_edges_": slice(120, -120),
        "lat_edges_": slice(30, 60),
        "center_lon": [210],
        "center_lat": [45],
        "color": "green"
    },
    "TA": {
        "ID": "TA",
        "long_name": "Tropical Atlantic",
        "lon_edges": slice(280, 30),
        "lat_edges": slice(90-30, 30),
        "lon_edges_": slice(-80, 30),
        "lat_edges_": slice(-30, 30),
        "center_lon": [335],
        "center_lat": [90],
        "color": "pink"
    },
    "IO": {
        "ID": "IO",
        "long_name": "Indian Ocean",
        "lon_edges": slice(30, 120),
        "lat_edges": slice(90-30, 30),
        "lon_edges_": slice(30, 120),
        "lat_edges_": slice(-30, 30),
        "center_lon": [75],
        "center_lat": [90],
        "color": "orange"
    },
    "TPO": {
        "ID": "TPO",
        "long_name": "Tropical Pacific",
        "lon_edges": slice(120, 280),
        "lat_edges": slice(90-30, 30),
        "lon_edges_": slice(120, -80),
        "lat_edges_": slice(-30, 30),
        "center_lon": [150, 210],
        "center_lat": [90],
        "color": "lime"
    },
    "SO": {
        "ID": "SO",
        "long_name": "Southern Ocean",
        "lon_edges": slice(0, 360),
        "lat_edges": slice(90-60, 90-30),
        "lon_edges_": slice(-180, 180),
        "lat_edges_": slice(-60, -30),
        "center_lon": [0],
        "center_lat": [90-45],
        "color": "yellow"
    },
    "AR": {
        "ID": "AR",
        "long_name": "Arctic",
        "lon_edges": slice(0, 360),
        "lat_edges": slice(60, 90),
        "lon_edges_": slice(-180, 180),
        "lat_edges_": slice(60, 90),
        "center_lon": [0],
        "center_lat": [75],
        "color": "deepskyblue"
    },
    "ANT": {
        "ID": "ANT",
        "long_name": "Antarctic",
        "lon_edges": slice(0, 360),
        "lat_edges": slice(0, 30),
        "lon_edges_": slice(-180, 180),
        "lat_edges_": slice(-90, -60),
        "center_lon": [0],
        "center_lat": [90-75],
        "color": "lime"
    }
}



import sys
import xarray as xr
import numpy as np
import os
from datetime import datetime, timedelta

# --- Define your configurations and zones here ---
# Ensure 'zones' dictionary is defined globally so the workers can see it
# zones = { ... } 
import xarray as xr
import numpy as np
import os
from datetime import datetime, timedelta

# Define Zones (Populate with your actual coordinates)
ZONES = {
    'NATLW': {'lon_edges': slice(280, 360), 'lat_edges': slice(30, 60)},
    'NATLE': {'lon_edges': slice(280, 360), 'lat_edges': slice(30, 60)},
    'NPW':   {'lon_edges': slice(140, 250), 'lat_edges': slice(30, 60)},
    'NPE':   {'lon_edges': slice(140, 250), 'lat_edges': slice(30, 60)},
}
EXP_NAME = "RT85_sst_0"
def get_composite_data(month_id):
    try:
        print(f"Starting Month {month_id}...",file=sys.stdout, flush=True)
        start_time = datetime.now()

        base_ds_dir = f"/home/philbou/scratch/isca_data/{EXP_NAME}/run{month_id:04d}/"
        curdir = f"{base_ds_dir}atmos_monthly.nc"
        curdir2 = f"{base_ds_dir}moist_data.nc"

        # 1. OPTIMIZATION: Use chunks to enable Dask (Lazy Loading)
        # This fixes the Memory 99% usage issue immediately.
        ds1 = xr.open_dataset(curdir, chunks={'time': 10},use_cftime=True)
        ds2 = xr.open_dataset(curdir2, chunks={'time': 10},use_cftime=True)
        
        # Merge is now instant (metadata only)
        comb_ds = xr.merge([ds1, ds2])

        lon_midlat_points = np.array([312, 345, 153, 226])
        lat_midlat_points = 45 * np.ones_like(lon_midlat_points)
        midlat_points_names = ['NATLW', "NATLE", 'NPW', 'NPE']

        # 2. OPTIMIZATION: .load() the tiny subset for math
        # We only bring these 4 grid points into RAM.
        ds_points = ds1.sel(
            lat=lat_midlat_points,
            lon=lon_midlat_points,
            method="nearest"
        ).load()

        # Math (Pure Numpy now, very fast)
        P_points_daysum = ds_points.precipitation.resample(time="1D").sum()
        P_along_pts = 86400 * P_points_daysum
        max_P_along_pts_id = P_along_pts.argmax(dim="time")[0, :]

        t_init = ds_points.time.values[0] + timedelta(hours=9)
        id_times = max_P_along_pts_id.values

        # 3. Write Data
        for i in range(len(midlat_points_names)):
            zone_name = midlat_points_names[i]
            zone = ZONES[zone_name]
            
            max_P_cft_time = t_init + timedelta(days=int(id_times[i]))
            
            # Lazy Slice (still no heavy data loaded)
            ds_select = comb_ds.sel(
                lon=zone["lon_edges"],
                lat=zone["lat_edges"],
                lat_interp=zone["lat_edges"],
                time=slice(-timedelta(days=0.5) + max_P_cft_time, 
                        timedelta(days=0.5) + max_P_cft_time)
            )

            outfile = f"{base_ds_dir}lon{lon_midlat_points[i]}.nc"
            if os.path.exists(outfile):
                os.remove(outfile)
            # This is the ONLY time heavy I/O happens
            ds_select.to_netcdf(outfile)

        # Cleanup
        ds1.close()
        ds2.close()
        
        end_time = datetime.now()
        print(f"Finished Month {month_id} in {end_time - start_time}",file=sys.stdout, flush=True)
        return f"Month {month_id}: Success"

    except Exception as e:
        return f"Month {month_id}: Failed ({e})"

def open_experiment(exp_folder_name, start_file, end_file, file_name, base_dir="/home/philbou/scratch/isca_data"):
    base = Path(base_dir) / exp_folder_name
    
    # Efficiently create file list
    files = [
        base / f"run{m:04d}" / file_name 
        for m in range(start_file, end_file + 1)
    ]

    # Quick check for missing files
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise EOFError(f'EXITING BECAUSE OF MISSING FILES: {missing}')

    ds = xr.open_mfdataset(
        files, 
        combine='nested', 
        concat_dim='time',
        decode_times=False,
        parallel=False,  # <--- Change this to False
    )
    return ds

def process_zone(zone_data):
    zid, lon_val = zone_data
    print(f"Processing zone {zid} (lon: {lon_val})",file=sys.stdout, flush=True)
    
    ds = open_experiment("RT42_sst_0_bucket", 301, 600, f"lon{lon_val}.nc")
    output_path = Path(f"data/lon{lon_val}_cumul.nc")
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
        # xarray's to_netcdf will overwrite by default if you don't specify mode='w'
    # but manually removing is safer if the file is corrupted
    if output_path.exists():
        output_path.unlink()
    
    ds.to_netcdf(output_path)
    print(f"Finished {zid}",file=sys.stdout, flush=True)

def add_season_coords(ds):
    """
    Adds a 'season' coordinate to a dataset with raw 360-day time units.
    Assumes time is in 'days since...' format.
    """
    # 1. Calculate the Month Index (0=Jan, 1=Feb, ... 11=Dec)
    #    % 360 wraps it to a single year
    #    // 30 drops it into 12 buckets of 30 days
    month_index = (ds.time % 360) // 30
    
    # 2. Map Month Index to Season Strings using numpy 'select'
    #    This is much faster than a loop
    conditions = [
        (month_index == 11) | (month_index <= 1),  # DJF (Dec, Jan, Feb)
        (month_index >= 2) & (month_index <= 4),   # MAM (Mar, Apr, May)
        (month_index >= 5) & (month_index <= 7),   # JJA (Jun, Jul, Aug)
        (month_index >= 8) & (month_index <= 10)   # SON (Sep, Oct, Nov)
    ]
    
    choices = ['DJF', 'MAM', 'JJA', 'SON']
    
    # Create the season array
    season_array = np.select(conditions, choices, default='Unknown')
    
    # 3. Add as a coordinate so we can use it for grouping
    ds.coords['season'] = (('time'), season_array)
    
    return ds

def save_seasonal(exp_name = "RT42_sst_0_bucket"):
    ds = open_experiment("RT42_sst_0_bucket", 301, 360, f"moist_data.nc")
    ds = add_season_coords(ds)

    # 3. Now you can calculate seasonal means in one line
    seasonal_means = ds.groupby('season').mean()
    seasonal_means.to_netcdf("data/seasonal_moist.nc")

# --- Processing Logic ---
z_names = ['NATLW', "NATLE", 'NPW','NPE']

lon_midlat_points = np.array([312,345,153,226])


if __name__ == "__main__":

    
# NUMBER OF WORKERS: Set this to match your SLURM --cpus-per-task
    N_WORKERS = 6
    
    print(f"Starting pool with {N_WORKERS} workers...", flush=True)

    # ProcessPoolExecutor replaces Dask. It is simpler and more stable.
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # Submit all jobs
        futures = {executor.submit(get_composite_data, m): m for m in range(61, 96)}
        
        # Monitor as they finish in real-time
        for future in as_completed(futures):
            result = future.result()
            print(result, flush=True)

    """zones = list(zip(z_names, lon_midlat_points))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # Submit all jobs
        futures = {executor.submit(process_zone, z): z for z in zones}
        
        # Monitor as they finish in real-time
        for future in as_completed(futures):
            result = future.result()
            print(result, flush=True)   """         
    print("All months processed.")
