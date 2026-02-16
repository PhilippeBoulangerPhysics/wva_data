#! /home/philbou/miniconda3/envs/pro_env python 
import xarray as xar
import os
from matplotlib.colors import LinearSegmentedColormap,Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as tkr
import cartopy.crs as ccrs
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts are quite similar to LaTeX's default font
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 15
plt.rcParams['legend.fontsize'] = 15
import sys
import numpy as np
import pdb
import time
from dask.diagnostics import ProgressBar
import cftime
from matplotlib.ticker import ScalarFormatter

yt = [-75,-50,-25,0,25,50,75]
blues = ["darkblue","dodgerblue","cornflowerblue","aliceblue"]
greens = ["darkgreen","mediumseagreen","limegreen","honeydew"]
pinks = ["crimson","deeppink","hotpink","lavenderblush"]
linestyles = ["solid","dashdot","dotted"]

R = 6371e3 #m
g = 9.8 #m/s**2




def split_save_exp(ds, exp_save_name):
    diag_groups = {
        "age": ['precip_age','sphum_age_1','sphum_age_2','sphum_age_3','sphum_age_4','sphum','precipitation','dt_sink',
                'height','phalf','ps','latb','lonb','dt_qg_convection','dt_qg_condensation','condensation_rain','convection_rain'],
        "dynamics": ['ps','bk','pk','sphum','ucomp','vcomp','omega','height','temp','vor','div'],
        "mixed_layer" : ['t_surf','flux_lhe', 'flux_t', 'flux_oceanq', 'corr_flux'],
        "atmosphere": ['precipitation','cape','rh'],
        "rrtm_rad": ['olr','toa_sw','tdt_rad','tdt_sw_rad','tdt_lw_rad','flux_sw'],
        "two_stream": ['olr','swdn_sfc', 'swdn_toa', 'lwup_sfc', 'lwdn_sfc','net_lw_surf', 'flux_rad', 'flux_lw', 'flux_sw']
    }

    default_dir = "/home/philbou/projects/def-rfajber/philbou/saved_ds"
    folder_path = os.path.join(default_dir, exp_save_name)
    os.makedirs(folder_path, exist_ok=True)

    print(f"Saving datasets to: {folder_path}")

    for group_name, variable_list in diag_groups.items():
        existing_vars = [var for var in variable_list if var in ds.variables]
        if not existing_vars:
            continue  # Skip empty groups

        subset = ds[existing_vars]

        file_path = os.path.join(folder_path, f"{group_name}.nc")

        # Use NetCDF4 with compression
        encoding = {
            var: {"zlib": True, "complevel": 4}
            for var in existing_vars
        }
        if group_name != "age":
            # Use Dask to write efficiently
            subset.to_netcdf(
                file_path,
                mode="w",
                format="NETCDF4",
                encoding=encoding,
                compute=True  # Forces execution, good for Dask-backed datasets
            )

            print(f"Saved: {file_path}")
        else:
            age_ds = subset
    return age_ds



def get_pot_temp(temp,pfull,ps):
    k = 0.286
    theta = np.zeros_like(temp)
    for i in range(len(pfull)):
        t2d = temp[:,i,:,:]
        tmp = t2d*(ps/(100*pfull[i]))**k
        theta[:,i,:,:] = tmp
    return theta
    
def decompose_time(a_total):
    """decompose into into a = <a> + a' """
    a = a_total
    a_avg = a_total.mean(dim = "time")
    a_prime = a-a_avg
    return [a_avg, a_prime,a]

def get_lapse_rate(pfull,phalf,temp):
    g=9.81
    R = 287 # Si units
    dP = phalf[1:] - phalf[:-1]
    dz = - (1/(g*pfull[:, np.newaxis, np.newaxis])) * R * temp* dP[:, np.newaxis, np.newaxis] #m
    dT = temp[1:,:,:] - temp[0:-1,:,:]
    atm_lapse_rate = -dT/dz[:-1] * 1000 # K/km
    atm_l_zonal = np.mean(atm_lapse_rate,axis = 2)
    return atm_l_zonal

def get_trop_p(col_lr_f,P_f):
    #plt.plot(col, P_f[1:]/100,marker = ".")
    #plt.ylim(1000,-10)
    #plt.vlines(2,0,1000,color = "red")
    mask = col_lr_f < 2
    #print(mask)
    i = 0
    for val in mask:
        if val:
            cur_P = P_f[i]/100
            #print(i,cur_P)
            if cur_P < 700:
                #print(i)
                return (cur_P + P_f[i-1]/100)/2
            else: continue
            
        i +=1

def tropopause_height(pfull_,phalf_,temp,lat):
    phalf = 100 * phalf_
    pfull = 100 * pfull_
    atm_l_zonal = get_lapse_rate(pfull,phalf,temp)
    trop_p = np.zeros_like(lat)
    P_f = np.flip(pfull)
    for i in range(len(lat)):
        col_lr = atm_l_zonal[:,i]
        col_lr_f = np.flip(col_lr)
        val_P = get_trop_p(col_lr_f,P_f)
        trop_p[i] = val_P
    return trop_p

def get_transport(value,uvw): 
    # calculate bar(bar(u)bar(val)) 
    
    value_decomp = decompose_time(value)
    valuebar = value_decomp[0]
    valueprime = value_decomp[1]
    
    transport = {}
    
    for key in ["u","v","omega"]:
        transport[key] = {}
        comp = uvw[key]
        ubar = comp["avg"]
        uprime = comp["prime"]
        mean_term = ubar*valuebar
        eddy_term = (uprime*valueprime).mean(dim = "time")
        total_term = mean_term+eddy_term
        transport[key]["mean"] = mean_term
        transport[key]["eddy"] = eddy_term
        transport[key]["total"] = total_term
    return transport

def get_p_ind(pval,pfull):

    difference_array = np.absolute(pfull-pval)

    # find the index of minimum element from the array
    index = difference_array.argmin()
    return index, pfull[index]

def scan_2d(diag,psurf,pval):
    tmp = diag.where(psurf >= pval, np.nan)
    return tmp


def plot_2d(diag,lat,lon,ax,cmap,level_space=25,land_mask = None,land_bool=False,lat_mask = None,proj = None,extend = "max"):
    """Plots any diagnostics in 2d """
    new_column = diag[:, [0]].copy()  # Avoiding data repetition at boundaries
    diag = np.hstack((diag, new_column))  # Handle wrap-around issue
    lon_mod = np.append(lon, lon[0] + 360)  # Fix longitude wrapping
    cs=ax.contourf(lon_mod, lat, diag,cmap=cmap,
                    extend=extend,levels = level_space,transform = proj)
    cbar = plt.colorbar(cs,shrink=0.5,orientation='horizontal',pad=0.01)
    for label in cbar.ax.get_xticklabels():
        label.set_rotation(45)  # or 30, 60, etc.
        label.set_ha('right')
    
    if land_bool:
        ax.contour(lon, lat_mask, land_mask,
                colors="black",levels = 1,transform = proj,linewidths=0.5)
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel("lon [deg]",fontsize = 24)
    ax.set_ylabel("lat [deg]",fontsize = 24)
    return cbar
def plot_vertical_profile(diag,lat,pvals,ax,cmap,mask,level_space=25,contours=False,lnP = False,
                          theta = False,thetac="",pot_temp_zonal=None,extend = "max",norm = False):
    """Plots any diagnostics in its vertical profile """
    diag = diag.where(~mask)
    if lnP: 
        #pvals = np.log(pvals)
        ax.set_yscale("log")
    
    if contours:
        print("contours",thetac)
        cs=ax.contour(lat, pvals, diag,colors=thetac,
                    extend=extend,levels = level_space)
        plt.clabel(cs, inline=True, fontsize=14)

    else:
        if norm == False:
            cs=ax.contourf(lat, pvals, diag,cmap=cmap,
                        extend=extend,levels = level_space)
        else:
            cs=ax.contourf(lat, pvals, diag,cmap=cmap,
                        extend=extend,norm = norm,levels = level_space)
            
        cbar = plt.colorbar(cs,shrink=0.8,orientation='vertical',pad=0.01)
        cbar.ax.tick_params(labelsize=20)
        
        #cbar.set_ticks(level_space) 
    if theta:
        print("here")
        isentropic_levels = np.arange(260, 400, 10)
        plot_vertical_profile(pot_temp_zonal,lat,pvals,ax,cmap = "thetac",thetac=thetac,level_space=isentropic_levels,contours=True)
    
    ax.set_xlabel("Latitude [deg]", fontsize = 24)
    ax.set_ylabel("Pressure [hPa]", fontsize = 24)
    ax.tick_params(axis='x', labelsize=18)  # Change font size for x-axis tick labels
    ax.tick_params(axis='y', labelsize=18)
    return cbar


def vertical_int(f,phalf,ps):
    dP = phalf[1:] - phalf[:-1]
    Ntime,Np,Nlat,Nlon = f.shape
    integral = np.zeros((Ntime,Nlat,Nlon))
    for i in range(Np):
        curP = phalf[i]
        mask_ps = ps>= curP 
        integral += dP[i] * f[:,i,:,:]* mask_ps[np.newaxis,:,:]
    integral *= 1/g
    return integral

def vertical_int_moist(f,q,bk, ps):
    tmp = vertical_int(f * q, bk, ps)
    tmp2 = vertical_int(q, bk, ps)
    return tmp/tmp2

def vertical_rho_avg(f, bk,ps):
    return vertical_int(f, bk,ps)/vertical_int(np.ones_like(f), bk,ps)

def age_precip(tau,dq):
    # For a given column.
    # first point is the age at TOA
    # This assumes that dq < 0 at TOA
    prev_age = 0
    cur_mu = 0
    prev_dq = dq[0]
    dq_l = prev_dq
    n_plev = len(tau)
    for i in range(1,n_plev):
        cur_age = tau[i]
        cur_dq = dq[i]
        cur_dq_l = cur_dq + dq_l
        if cur_dq < 0: # precipitation
            cur_mu = (cur_age * cur_dq + dq_l * cur_mu)/(cur_dq_l)

        elif cur_dq >= 0:
            pass

            # dq_l gets updated anyway
            #prev_mu stays the same
        if cur_dq_l > 0: dq_l=0
        else: dq_l = cur_dq_l

    return cur_mu


def get_age_precip(ds):

    # Use Dask-aware arrays

    qT = ds.sphum_age_1
    q = ds.sphum
    tau = qT / q

    dq_conv = ds.dt_qg_convection
    dq_cond = ds.dt_qg_condensation
    P_cond = ds.condensation_rain  # already in kg/kg/s
    P_conv = ds.convection_rain

    # Efficient parallel apply using Dask
    p_age_cond = xar.apply_ufunc(
        age_precip, tau, dq_cond,
        input_core_dims=[["pfull"], ["pfull"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[tau.dtype]
    )

    p_age_conv = xar.apply_ufunc(
        age_precip, tau, dq_conv,
        input_core_dims=[["pfull"], ["pfull"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[tau.dtype]
    )

    # Avoid divide-by-zero
    total_precip = P_conv + P_cond
    p_age = xar.where(total_precip > 0,
                     (p_age_cond * P_cond + p_age_conv * P_conv) / total_precip,
                     0.0)

    return p_age



def get_area(tmp_ds):
    # Calculate the area 

    lat = np.deg2rad(tmp_ds.lat.values)
    latb = tmp_ds.latb.values
    lon = tmp_ds.lon.values
    lonb = tmp_ds.lonb.values

    dlat = np.deg2rad(latb[1:] - latb[:-1])
    dlon = np.deg2rad(lonb[1:] - lonb[:-1])
    R = 6371
    area = np.zeros((len(lat),len(lon)))

    for i in range(len(lat)):
        for j in range(len(lon)):
            dlat_tmp = dlat[i]
            dlon_tmp = dlon[j]
            lat_tmp = lat[i]

            tmp = R**2 * np.cos(lat_tmp) * dlon_tmp * dlat_tmp

            area[i][j] = abs(tmp)
    return area

def area_w_avg(area,t_surf, D3 = False):
    temp = 0
    A = 0
    if D3:
        nlat = t_surf.shape[1]
        nlon = t_surf.shape[2]
    else:
        nlat = t_surf.shape[0]
        nlon = t_surf.shape[1]
    for i in range(nlat):
        for j in range(nlon):
            if D3:
                t = t_surf[:,i,j]
            else:
                t = t_surf[i][j]
            a = area[i][j]
            temp += a*t
            A += a
    glob_avg_temp = temp/A
    return glob_avg_temp


def plot_age_moments_2d(pval,max_val,mom,n_moment,ps,lat,lon,pfull,fig, ax, time_avg = True,cmap = "gist_ncar",title = "Default Title",land_bool = False,land_mask = "None",min_val=0,steps =41,ext = "max",scan = True):
    plevel,pvall = get_p_ind(pval,pfull)
        
    mom_p = mom.isel(pfull = plevel)

    if time_avg:
        mom_p_t = mom_p.mean(dim="time")
        psurf = ps.mean(dim="time")
    else:
        mom_p_t = mom_p.isel(time = -1)
        psurf = ps.isel(time = -1)
        
    mom_p_t *= 1/(60*60*24)**n_moment
    mom_p_t = mom_p_t**(1/n_moment)
        
    if scan :
        mom_p_t = scan_2d(mom_p_t,psurf/100,pvall)
    
    cbar = plot_2d(mom_p_t.values,lat,lon,ax,cmap,level_space=np.linspace(min_val,max_val,steps),land_bool=land_bool,land_mask=land_mask,proj = ccrs.PlateCarree(),extend=ext)
    ax.gridlines()
    ax.set_title(title,fontsize = 35)
    return fig,ax,cbar
    
def plot_shape_param_2d(pval,max_val,shape,ps,lat,lon,pfull,fig, ax,time_avg = True,file_save= True,cmap = "gist_ncar",title = "Default Title",land_bool = False,land_mask = "None",min_val = 0,steps = 41,ext = "max",scan = True):
    plevel,pvall = get_p_ind(pval,pfull)

    mom_p = shape.isel(pfull = plevel)
    if time_avg:
        mom_p_t = mom_p.mean(dim="time")
        psurf = ps.mean(dim="time")
    else:
        mom_p_t = mom_p.isel(time = -1)
        psurf = ps.isel(time = -1)
        
        
    if scan: mom_p_t = scan_2d(mom_p_t,psurf/100,pvall)
    
    
    cbar = plot_2d(mom_p_t.values,lat,lon,ax,cmap,level_space=np.linspace(min_val,max_val,steps),land_bool=land_bool,land_mask=land_mask,proj = ccrs.PlateCarree(),extend=ext)
    ax.gridlines()
    ax.set_title(title)
    return fig,ax,cbar 

def get_stream_fct(uvw,pfull,phalf,lat, time_avg = True):
    if time_avg: v = uvw["v"]["avg"]
    else: v = uvw["v"]["prime"].mean(dim = "time")
    
    vz = v.mean(dim = ["lon"]).values
    vert_int = vertical_int(pfull,phalf,vz,0,1000)
    psi = 2 * np.pi * R * np.cos(np.deg2rad(lat)) * vert_int / g
    return psi

def plot_stream_fct(stream_fct_timeavg,stream_fct_timeprime,pval,lat,lnP=False,cmap = "seismic"):
    fig,axs = plt.subplots(1,2,figsize=(21,5))
    names = [r"$\overline{\psi}$",r"$\psi'$"]
    streams = [stream_fct_timeavg,stream_fct_timeprime]
    i=0
    for stream in streams:
        # get the greater of min and max
        vmax = stream.max()
        vmin = stream.min()
        limit = max(np.abs(vmin), np.abs(vmax))
        
        if limit >0:levelspace = np.linspace(-limit,limit,30)
        else: levelspace=25
        
        plot_vertical_profile(stream,lat,pval,axs[i],cmap,level_space=levelspace,lnP=lnP)
        axs[i].set_title(names[i])
        i+=1
    
    fig.suptitle(r"Stream function : $\psi$")
    
    return fig,axs

def plot_transport_vert(transport,var_str,pval,lat,lev=20,cmap="gist_ncar",typ = "",lnP=False,latlim = (-90,90)):
    lim1,lim2 = latlim
    # Vertical
    fig, ax = plt.subplots(1,1,figsize = (10,10)) 
    uq = transport.mean(dim = "lon").values
    title = f"transport {var_str}"
    ax.set_title(title)
    plot_vertical_profile(uq,lat,pval,ax,cmap,level_space=lev,lnP=lnP,extend="both")
    
    ax.set_xlim(lim1,lim2)
    
    return fig, ax
    
def plot_transport_2d(transport,pval,pfull,lat,lon,var_str,lev = 30,typ = "",cmap = "gist_ncar",land_bool = False,land_mask = "None"):
    
    plevel,pvall = get_p_ind(pval,pfull)
    
    transportt = transport.isel(pfull = plevel)
        
    fig, ax = plt.subplots(nrows=1, ncols=1, 
                            subplot_kw={'projection': ccrs.Robinson()},figsize = (15,10))
    
    
    title = f"transport {var_str} at P= {round(pvall,2)} hPa"
    
    plot_2d(transportt.values,lat,lon,ax,cmap,level_space=lev,land_bool=land_bool,land_mask=land_mask,proj = ccrs.PlateCarree(),extend = "both")
    ax.gridlines()
    ax.set_title(title)
    
    return fig,ax
    
def plot_circulation_uvw(uvw,pval,lat,
                        time_avg = True,lnP=False):
    
    if time_avg:  
        title = "Circulation: Zonal and time avg"     
        u_= uvw["u"]["avg"]
        v_= uvw["v"]["avg"]
        w_= uvw["omega"]["avg"]
    else:
        title = "Circulation: Zonal avg"  
        u_= uvw["u"]["prime"].mean(dim = "time")
        v_= uvw["v"]["prime"].mean(dim = "time")
        w_= uvw["omega"]["prime"].mean(dim = "time")
    
    u_ = u_.mean(dim = "lon")
    v_ = v_.mean(dim = "lon")
    w_ = w_.mean(dim = "lon")
    
    total = [u_,v_,w_]
    
    
    fig,axs = plt.subplots(1,3,figsize=(30,8))
    
    for i in range(3):
        cur = total[i]
        max_u = np.max(abs(cur))
        lev_u= np.linspace(-abs(max_u),abs(max_u),30)
        #print(lev_u)
        plot_vertical_profile(cur,lat,pval,axs[i],level_space=lev_u,cmap = "seismic",lnP=lnP)

    if time_avg: 
        subtitle = [r"$\overline{u}$",r"$\overline{v}$" ,r"$\overline{\omega}$" ]
    else: subtitle = [r"$u'$",r"$v'$",r"$\omega'$"]
    
    for i in range(3):
        axs[i].set_title(subtitle[i])
        
    fig.suptitle(title)
    return fig,axs
    
def plot_age_moments_vertical_profile(max_val,mom,n_moment,ps,pfull,lat,mask,pot_temp="None",central = False,time_avg = True,
                                        file_save= True,cmap = "Blues",lnP = False,theta = True,custom = False,
                                        ax = "None",fig = "None",min_val = 0,steps = 29,title = "default_title",col = "black",ext = "max"):

    if time_avg:
        mom_t = mom.mean(dim="time")
        psurf = ps.mean(dim="time")
    else:
        mom_t = mom.isel(time = -1)
        psurf = ps.isel(time = -1)
        
    mom_t *= 1/(60*60*24)**n_moment

    mom_t = mom_t**(1/n_moment)
    
    for i in range(len(pfull)):
        cur_mom = mom_t.isel(pfull = i).copy()
        cur_mom = scan_2d(cur_mom,psurf/100,pfull[i])
        mom_t.isel(pfull = i).values = cur_mom
        
    mom_z_t = mom_t.mean(dim = "lon")

    if custom == False:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (9,6))
    
    cbar = plot_vertical_profile(mom_z_t,lat,pfull,ax,cmap,mask,level_space=np.linspace(min_val,max_val,steps),lnP = lnP,theta = False,thetac = col,extend = ext)
    if theta:
        isentropic_levels = np.arange(260, 400, 10)
        cs=ax.contour(lat, pfull, pot_temp,colors=col,levels = isentropic_levels)
        plt.clabel(cs, inline=True, fontsize=14)    
    return fig,ax,cbar
    

    
def plot_shape_param_vertical_profile(shape,pfull,lat,mask,ps,pot_temp,max = 2,lnP=False,central = False,
                                        time_avg = True,file_save= True,cmap = "Blues"
                                        ,custom = False, ax = "None",fig = "None"):
    mom = shape
    
    if time_avg:
        mom_t = mom.mean(dim="time")
        psurf = ps.mean(dim="time")
    else:
        mom_t = mom.isel(time = -1)
        psurf = ps.isel(time = -1)
    
    for i in range(len(pfull)):
        cur_mom = mom_t.isel(pfull = i).copy()
        cur_mom = scan_2d(cur_mom,psurf/100,pfull[i])
        mom_t.isel(pfull = i).values = cur_mom
        
    mom_z_t = mom_t.mean(dim = "lon")
    if custom == False:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (9,6))
    
    ext = ""
    if time_avg: ext = "_avg"
    ext2=""
    
    if lnP: 
        ext2="_log"
    title = r"Zonal Average of Shape Parameter $s$"
    tmp_folder = f"/shape_param"

    plot_vertical_profile(mom_z_t,lat,pfull,ax,cmap,mask,level_space=np.linspace(0,max,41),lnP = lnP)
    isentropic_levels = np.arange(260, 400, 10)
    cs=ax.contour(lat, pfull, pot_temp,colors="black",levels = isentropic_levels)
    plt.clabel(cs, inline=True, fontsize=14)

    return fig,ax


def plot_sphum_2d(sphum, pval,pfull,lat,lon,time_avg = True,file_save= True,cmap = "Blues",land_bool = False,land_mask = "None"):
    plevel,pvall = get_p_ind(pval,pfull)
    if time_avg:
        sphum_t = sphum.mean(dim="time")
        title = r"Specific Humidity $[kg kg^{-1}]$"
    else:
        sphum_t = sphum.isel(time = -1)
    
    sphum_t_p = sphum_t.isel(pfull = plevel)
        
    fig, ax = plt.subplots(nrows=1, ncols=1, 
                            subplot_kw={'projection': ccrs.Robinson()},figsize = (15,10))
    
    
    title = f"specific humidity [kg/kg] at P= {round(pvall,2)} hPa"
    
    plot_2d(sphum_t_p.values,lat,lon,ax,cmap,level_space=30,land_bool=land_bool,land_mask=land_mask,proj = ccrs.PlateCarree())
    ax.gridlines()
    ax.set_title(title)
    
    return fig,ax

def plot_sphum_vert(sphum,pfull,lat,lev = 25,time_avg = True,file_save=True,cmap = "Blues",lnP=False):
    """ Saves horizontal profile of precipitation """
    if time_avg:
        sphum_t = sphum.mean(dim="time")
    else:
        sphum_t = sphum.isel(time = -1)
    
    
    title = r"Specific Humidity $[kg kg^{-1}]$"
        
    # plot vertical profile
    zonal_sphum_t = sphum_t.mean(dim = "lon")
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize = (15,10))
    ax.set_title(title)
    plot_vertical_profile(zonal_sphum_t,lat,pfull,ax,cmap,level_space=lev,lnP=lnP)

    return fig,ax

def plot_precipitation(precipitation, lat,lon,ax,lev,time_avg = True,file_save=True,cmap = "Blues",tit= None,land_bool = False,land_mask = "None"):
    """ Saves horizontal and vertical profile of specific humidity """
    if time_avg:
        precipitation_t = precipitation.mean(dim="time")
    else:
        precipitation_t = precipitation.isel(time = -1)
    if tit == None:  
        title = r"Precipitation $[g \cdot kg^{-1} s^{-1}]$"
    else: title = tit
    fig, ax = plt.subplots(nrows=1, ncols=1, 
                            subplot_kw={'projection': ccrs.Robinson()},figsize = (15,10))    
    plot_2d(1000*precipitation_t.values,lat,lon,ax,cmap,land_bool=land_bool,land_mask=land_mask,level_space=lev,proj = ccrs.PlateCarree())
    ax.gridlines()
    ax.set_title(title)

    return fig, ax
    
    
def plot_vert_int_age(qage,sphum,max_val,lat,lon,phalf,time_avg = True,file_save= True,cmap = "gist_ncar",title = "Default Title",land_bool = False,land_mask = "None"):
    v_age = vertical_int(qage,phalf)
    v_age = v_age / (vertical_int(sphum.values,phalf))
    v_age_time = np.mean(v_age,axis =0)/(24*60**2)

    fig, ax = plt.subplots(nrows=1, ncols=1, 
                                subplot_kw={'projection': ccrs.Robinson()},figsize = (15,10))
    plot_2d(v_age_time,lat,lon,ax,cmap="gist_ncar",level_space=np.linspace(0,14,43),land_bool=land_bool,land_mask=land_mask,proj = ccrs.PlateCarree())
    ax.gridlines()
    ax.set_title(title)
    return fig,ax


def add_precip_to_diag(exp_folder_name, exp_save_name,start_file, end_file, file_name = "atmos_monthly.nc"):
    path_to_files = f"/home/philbou/scratch/isca_data/{exp_folder_name}/"
    
    for id_file in np.arange(start_file,end_file,1):
        print(id_file)
        ds_path = f"{path_to_files}run{id_file:04d}/{file_name}"
        print("loading dataset")
        ds = xar.open_mfdataset(ds_path,parallel=True)
        print("strat precip calculation")
        p_age = get_age_precip(ds)
        print("done precipitation age")
        ds["precip_age"] = xar.DataArray(
                p_age,
                dims=("time","lat", "lon"),
                coords={"time":ds.time, "lat": ds.lat, "lon": ds.lon},
                attrs={"units": "sec"}  # or whatever units you're using
            )
        print("saving to netcdf")
        ds.to_netcdf(ds_path,mode = "w")
        ds.close()  # Clean up
        del ds
        print("done!")


def procress_climate_run(exp_folder_name, exp_save_name, start_file, end_file, file_name="atmos_monthly.nc"):
    print("Step 1: Loading dataset with Dask")
    ds = open_experiment(exp_folder_name, start_file, end_file, file_name)  # must return Dask-backed Dataset
    print("Dataset loaded")

    print("Step 2: Calculating precipitation age")
    ds["precip_age"] = get_age_precip(ds)  # should be Dask-aware already
    print("Precipitation age added")

    print("Step 3: Computing monthly mean (lazy)")
    # Use Dask-aware resampling
    n_steps = ds.dims["time"]
    steps_per_month = 120
    n_months = n_steps // steps_per_month

    # Trim to full months only
    ds_trimmed = ds.isel(time=slice(0, n_months * steps_per_month))
    ds_trimmed = ds_trimmed.chunk({'time': steps_per_month})  # Match coarsen window
    monthly_ds = ds_trimmed.coarsen(time=steps_per_month, boundary="trim").mean()
    
    print("finally m,aking dask calculate stuff")
    monthly_ds = monthly_ds.persist()
    print("Now saving")
    default_dir = "/home/philbou/projects/def-rfajber/philbou/saved_ds"
    folder_path = os.path.join(default_dir, exp_save_name)
    os.makedirs(folder_path, exist_ok=True)
    
    with ProgressBar():
        monthly_ds.to_netcdf(f"{default_dir}/{exp_save_name}/entire_dataset.nc")
    print("Step 4: Saved dataset as a whole")
    """  # Show progress for long saves
    with ProgressBar():
        split_save_exp(monthly_ds, exp_save_name)
    """
    print("All groups saved to disk: don't forget to save the restart and input as well.")
    