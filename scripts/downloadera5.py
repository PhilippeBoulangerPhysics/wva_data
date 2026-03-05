#!/usr/bin/env python
import cdsapi

c = cdsapi.Client()

dataset = "reanalysis-era5-complete"

request = {
    "class": "ea",
    "date": "2005-01-01/to/2019-12-31",
    "expver": "1",
    "levelist": "1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/800/850/900/925/950/975/1000",  # Pressure levels in hPa
    "levtype": "pl",  # Changed from 'pv' to 'pl' for pressure levels
    "param": "133.128",  # Specific humidity parameter code
    "stream": "oper",
    "time": ['00:00', '06:00', '12:00', '18:00'],
    "type": "moda",  # Changed from 'an' to 'moda' for monthly means
    'area': '90/-180/-90/180',  # Global coverage
    'grid': '0.25/0.25',
    "data_format": "netcdf",
}

output = 'sphum_monthly_2005_2019.nc'

c.retrieve(dataset, request, output)