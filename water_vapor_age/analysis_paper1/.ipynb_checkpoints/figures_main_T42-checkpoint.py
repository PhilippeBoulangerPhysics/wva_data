import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os


os.chdir("/home/philbou/projects/def-rfajber/philbou/analysis_paper1")

import diagnostic_plot_helper as dps  
from scipy.special import gamma
import gc
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import ListedColormap
import sys

def get_global_2D(term,area):
    # term is (time,lat,lon)
    twoD_avg = dps.area_w_avg(area, term, D3=True) # (time)
    time_2D_avg = np.mean(twoD_avg,axis = 0)
    return time_2D_avg

def decomp_term_time_2D_vert(term,area,phalf,ps,temp):
    # term is (time,pfull,lat,lon)
    vert_avg = dps.vertical_int_z(term,temp, phalf,ps) # (time,lat,lon)
    vert_2D_avg = dps.area_w_avg(area, vert_avg, D3=True) # (time)
    time_vert_2D_avg = np.mean(vert_2D_avg,axis = 0)
    
    prime = term - time_vert_2D_avg
    return term,time_vert_2D_avg,prime

def decomp_term_time_2D(term,area):
    # term is (time,lat,lon)
    twoD_avg = dps.area_w_avg(area, term, D3=True) # (time)
    time_2D_avg = np.mean(twoD_avg,axis = 0)
    
    prime = term - time_2D_avg
    return term,time_2D_avg,prime

def decomp_term_time(term):
    # term is (time,pfull,lat,lon)

    time_avg = term.mean(dim = "time")
    
    prime = term - time_avg
    return term,time_avg,prime

def get_eddy_kin_energy(ucomp,vcomp):
    u,global_u,u_prime = decomp_term_time(ucomp)
    v,global_v,v_prime = decomp_term_time(vcomp)
    uterm = (u_prime**2).mean(dim = "time",keepdims=True)
    vterm = (v_prime**2).mean(dim = "time",keepdims=True)
    return 0.5* (uterm + vterm)

def gamma_dist(x,mean,mom2):
    theta = (mom2-mean**2)/(mean)
    alpha = mean/theta
    num = x**(alpha -1) * np.exp(-x/theta)
    den = gamma(alpha) * theta**alpha
    return num/den

def weibull_dist(x,mean,mom2):
    std = np.sqrt(mom2-mean**2)
    b_ = (mean/std)**1.086
    a_ = mean/(gamma(1+1/b_))
    return (b_/a_)*(x/a_)**(b_-1)*np.exp(-(x/a_)**b_)

def get_alpha_theta(mean,mom2):
    theta = (mom2-mean**2)/(mean)
    alpha = mean/theta
    return alpha, theta

def get_a_b(mean,mom2):
    std = np.sqrt(mom2-mean**2)
    b_ = (mean/std)**1.086
    a_ = mean/(gamma(1+1/b_))
    return a_, b_

def get_n_moment_weib(a,b,n):
    return a**n * gamma(1 + (n/b))

def get_n_moment_gamma(alpha, theta, n):
    return theta**n * gamma(alpha+n)/gamma(alpha)

path_cur = "/home/philbou/projects/def-rfajber/philbou/analysis_paper1/Figures_final"

land_mask_name = "era_land_t42.nc"
land_bool = True
if land_bool:
    lm_path = "~/Isca/exp/test_cases/realistic_continents/input/"+land_mask_name
    ds_landmask = xr.open_dataset(lm_path) 
    land_mask = ds_landmask.land_mask.values
    
plt.rcParams['text.usetex'] = True

# Load and process datasets
delta_sst_values = np.array([-2,0,2])
ds_dict = {}
i=0
for delta_sst in delta_sst_values:
    path_folder = "~/projects/def-rfajber/philbou/saved_ds/"
    #path_folder = "./"
    base_folder_name = "RT42_sst_"


    exp_folder_name = f"{base_folder_name}{'m' + str(abs(delta_sst)) if delta_sst < 0 else str(delta_sst)}_"
    path = f"{path_folder}/{exp_folder_name}"

    ds_age = xr.open_dataset(f"{path}/age.nc")
    ds_mix = xr.open_dataset(f"{path}/mixed_layer.nc",decode_times = False)
    ds_dyn = xr.open_dataset(f"{path}/dynamics.nc")

    ds_precip_age = xr.open_dataset(f"{path}/precipitation_age_2.nc")

    ds_rad = xr.open_dataset(f"{path}/rrtm_rad.nc")
    
    ds_age["time"] = np.arange(0,len(ds_age.time.values),1)
    ds_dyn["time"] = np.arange(0,len(ds_dyn.time.values),1)
    ds_rad["time"] = np.arange(0,len(ds_rad.time.values),1)
    ds_precip_age["time"] = np.arange(0,len(ds_precip_age.time.values),1)
    
    area = dps.get_area(ds_age)
    
    ts = ds_mix.t_surf.values
    time_global_ts = dps.area_w_avg(area,np.mean(ts, axis=0))
    temp = ds_dyn.temp
    q = ds_age.sphum
    sphum = ds_age.sphum
    qmoments = [ds_age.sphum_age_1,ds_age.sphum_age_2]
    moments = [qmoments[0]/sphum,qmoments[1]/sphum]
    central_moments = [moments[0], moments[1]-moments[0]**2]

    mean = moments[0]
    ps = ds_age.ps

    psint = ds_age.ps.mean(dim= "time").values
    pfull = ds_age.pfull
    phalf = 100*ds_age.phalf.values
    lat = ds_age.lat
    lon = ds_age.lon
    pot_temp = dps.get_pot_temp(temp,pfull,ps)
    pot_temp_zonal = np.mean(np.mean(pot_temp,axis = 0),axis = 2)
    std = central_moments[1]**(1/2)

    b = (mean/std)

    a = mean/(gamma(1+1/b))
    
    v_age = dps.vertical_int_moist(moments[0].values,q,phalf,psint)
    v_age_time = np.mean(v_age,axis =0)/(24*60**2)

    v_shape = dps.vertical_int_moist(b.values,q,phalf,psint)
    v_shape_time = np.mean(v_shape,axis =0)
    
    v_std = dps.vertical_int_moist(std.values,q,phalf,psint)
    v_std_time = np.mean(v_std,axis =0)
    
    precip_age = ds_precip_age.precip_age.mean(dim = "time")
    
    tropopause_height = dps.tropopause_height(pfull.values,phalf/100,temp.mean(dim = "time").values,lat.values)
    ts_global = get_global_2D(ts,area)
    
    
    tmp_dic = {
        'delta_sst_model': delta_sst,
        'ds_age': ds_age,
        'ds_dyn': ds_dyn,
        'ds_rad': ds_rad,
        'global_surf_temp': time_global_ts,
        "temp": temp,
        "q": q,
        "sphum": sphum,
        "qmoments": qmoments,
        "moments": moments,
        "central_moments": central_moments,
        "mean": mean,
        "ps": ps,
        "pfull": pfull,
        "phalf": phalf,
        "lat": lat,
        "lon": lon,
        "pot_temp": pot_temp,
        "pot_temp_zonal": pot_temp_zonal,
        "std": std,
        "shape": b,
        "a": a,
        "ts_global" : ts_global,
        "tropopause_height": tropopause_height,
        "precip_age" : precip_age,
        "vert_time_mean" : v_age_time,
        "vert_time_shape" : v_shape_time,
        "v_std_time" : v_std_time
    }

    ds_age_precip = xr.open_dataset(f"{path}/age_precip.nc")
    ds_age_precip["time"] = np.arange(0,len(ds_age_precip.time.values),1)
    tmp_dic['ds_age_precip'] = ds_age_precip
        
    ds_dict[str(delta_sst)] = tmp_dic
    del ds_dyn, ds_mix,delta_sst, ds_age, time_global_ts, temp, lon,q, sphum, qmoments, moments, central_moments, mean, ps, pfull, lat, pot_temp, pot_temp_zonal, std, b, a,v_shape_time,v_age_time
    gc.collect()


control_sst_val = ds_dict['0']['ts_global']
delta_sst_exact = ds_dict['2']['ts_global'] - control_sst_val

anomaly_mean = (ds_dict['2']["mean"]  - ds_dict['0']["mean"])/delta_sst_exact
anomaly_std = (ds_dict['2']["std"]  - ds_dict['0']["std"])/delta_sst_exact
anomaly_shape = (ds_dict['2']["shape"]  - ds_dict['0']["shape"])/delta_sst_exact
anomaly_vert_mean = (ds_dict['2']["vert_time_mean"]  - ds_dict['0']["vert_time_mean"])/delta_sst_exact
anomaly_vert_shape = (ds_dict['2']["vert_time_shape"]  - ds_dict['0']["vert_time_shape"])/delta_sst_exact

rel_anomaly_mean = 100*(ds_dict['2']["mean"]  - ds_dict['0']["mean"])/ds_dict['0']["mean"]/delta_sst_exact
rel_anomaly_std = 100*(ds_dict['2']["std"]  - ds_dict['0']["std"])/ds_dict['0']["std"]/delta_sst_exact
rel_anomaly_shape = 100*(ds_dict['2']["shape"]  - ds_dict['0']["shape"])/ds_dict['0']["shape"]/delta_sst_exact
rel_anomaly_vert_mean = 100*(ds_dict['2']["vert_time_mean"]  - ds_dict['0']["vert_time_mean"])/ds_dict['0']["vert_time_mean"]/delta_sst_exact
rel_anomaly_vert_shape = 100*(ds_dict['2']["vert_time_shape"]  - ds_dict['0']["vert_time_shape"])/ds_dict['0']["vert_time_shape"]/delta_sst_exact

mean = ds_dict['0']["mean"]
shape = ds_dict['0']["shape"]
std = ds_dict['0']["std"]
lat = ds_dict['0']["lat"]
lon = ds_dict['0']["lon"]
pot_temp_zonal = ds_dict['0']["pot_temp_zonal"]
pot_temp_zonal2 = ds_dict['2']["pot_temp_zonal"]
ps = ds_dict['0']["ps"]
pfull = ds_dict['0']["pfull"]

trop_h_0 = ds_dict["0"]["tropopause_height"]
trop_h_2 = ds_dict["2"]["tropopause_height"]
trop_h_m2 = ds_dict["-2"]["tropopause_height"]

sink0 = ds_dict["0"]["ds_age"].dt_sink
q0 = ds_dict["0"]["q"]
sink2 = ds_dict["2"]["ds_age"].dt_sink
q2 = ds_dict["2"]["q"]

#------------------------------------
fig,ax = plt.subplots(1,2,figsize = (22,8),gridspec_kw={'hspace': 0.3,'wspace': 0.05})

ax_sink = ax[0]
fig_,ax_sink = dps.plot_age_moments_vertical_profile(15,1e3*q0 * (24*60**2),1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "YlGnBu",
                                                    time_avg = True,lnP=False,custom = True, ax = ax_sink,steps=51,min_val = 0,ext = "max")
ax_sink.set_title("$q$ [$g\\cdot kg^{-1}$]\n       (A)", pad=20, fontsize=30)
ax_sink = ax[1]
fig_,ax_sink = dps.plot_age_moments_vertical_profile(5,-1e3*sink0 * (24*60**2)**2,1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "YlGnBu",
                                                    time_avg = True,lnP=False,custom = True, ax = ax_sink,steps=51,min_val = 0,ext = "max")
ax_sink.set_title("$S^-$ [$g\\cdot kg^{-1}\\cdot $day$^{-1}$]\n       (B)",pad = 20,fontsize = 30)
ax_sink.set_ylabel(" ")
for axi in ax:
    axi.set_ylim(975,200)
plt.savefig(f"{path_cur}/qsink.png",dpi = 250,bbox_inches = "tight")
plt.close()
print("Saved qsink.png",file=sys.stdout, flush=True)
#------------------------------------
fig,ax = plt.subplots(1,2,figsize = (22,8),gridspec_kw={'hspace': 0.3,'wspace': 0.05})

ax_sink = ax[0]
fig_,ax_sink = dps.plot_age_moments_vertical_profile(1.5,1e3*(q2-q0) * (24*60**2)/delta_sst_exact,1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "PuOr",
                                                    time_avg = True,lnP=False,custom = True, ax = ax_sink,steps=51,min_val = -1.5,ext = "neither",theta = True)

ax_sink.set_title("$q$ [$g\\cdot kg^{-1}\\cdot K^{-1}$]\n       (A)", pad=20, fontsize=30)
ax_sink = ax[1]
fig_,ax_sink = dps.plot_age_moments_vertical_profile(0.25,-1e3*(sink2-sink0) * (24*60**2)**2/delta_sst_exact,1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "PuOr",
                                                    time_avg = True,lnP=False,custom = True, ax = ax_sink,steps=51,min_val = -0.25,ext = "max")
ax_sink.set_title("$S^-$ [$g\\cdot kg^{-1}\\cdot $day$^{-1}\\cdot K^{-1}$]\n       (B)",pad = 20,fontsize = 30)
ax_sink.set_ylabel(" ")
for axi in ax:
    axi.set_ylim(975,200)
fig.suptitle("Anomally",fontsize = 35,y = 1.06)
plt.savefig(f"{path_cur}/qsink_anomaly.png",dpi = 250,bbox_inches = "tight")
plt.close()
print("Saved qsink_anomaly.png",file=sys.stdout, flush=True)
# -------------------------------------------------------

def get_stream_fct(v,w,phalf,ps,latb,lat):
    R = 6371
    lat_rad = np.deg2rad(lat)
    lat_radb = np.deg2rad(latb)
    dlat = lat_radb[1:]-lat_radb[0:-1]
    
    psi_v = np.zeros_like(v)
    for plev_i in range(len(phalf)-2):
        psi_v_i = dps.vertical_int(v[:,:plev_i+1,:,:],phalf[:plev_i+2],ps)
        psi_v[:,plev_i,:,:] = psi_v_i
    
    tmp = np.cos(lat_rad)
    return 2*np.pi * R * tmp[np.newaxis,np.newaxis,:,np.newaxis] * psi_v


ucomp0 = ds_dict['0']['ds_dyn'].ucomp
vcomp0 = ds_dict['0']['ds_dyn'].vcomp
omega0 = ds_dict['0']['ds_dyn'].omega
q0 = ds_dict['0']['ds_age'].sphum
latb = ds_dict['0']['ds_age'].latb
A0 = get_stream_fct(vcomp0.values,omega0.values,phalf,psint,latb.values,lat.values)
streamfct0 = np.mean(np.mean(A0,axis = 3),axis = 0)

qvcomp0 = vcomp0 * q0
ds_6h0 = ds_dict['0']['ds_age_precip']
q0 = ds_6h0.sphum
v0 = ds_6h0.vcomp
u0 = ds_6h0.ucomp
qprime0 = q0 - q0.mean(dim = ["time"])
vprime0 = v0 - v0.mean(dim = ["time"])
uprime0 = u0 - u0.mean(dim = ["time"])
vqprime0 = vprime0 * qprime0
kin0 =  0.5 *( v0**2 + u0**2)
eddykin0 = kin0 - kin0.mean(dim = ["time"])
eddykinprime0 = vprime0 * eddykin0

ucomp2 = ds_dict['2']['ds_dyn'].ucomp
vcomp2 = ds_dict['2']['ds_dyn'].vcomp
omega2 = ds_dict['2']['ds_dyn'].omega
q2 = ds_dict['2']['ds_age'].sphum
latb = ds_dict['2']['ds_age'].latb
A2 = get_stream_fct(vcomp2.values,omega2.values,phalf,psint,latb.values,lat.values)
streamfct2 = np.mean(np.mean(A2,axis = 3),axis = 0)

qvcomp2 = vcomp2 * q2
ds_6h2 = ds_dict['2']['ds_age_precip']
q2 = ds_6h2.sphum
v2 = ds_6h2.vcomp
u2 = ds_6h2.ucomp
qprime2 = q2 - q2.mean(dim = ["time"])
vprime2 = v2 - v2.mean(dim = ["time"])
uprime2 = u2 - u2.mean(dim = ["time"])
vqprime2 = vprime2 * qprime2
kin2 =  0.5 *( v2**2 + u2**2)
eddykin2 = kin2 - kin2.mean(dim = ["time"])
eddykinprime2 = vprime2 * eddykin2

# ---------------------------------------------------------
fig, ax = plt.subplots(2,2,figsize = (22,16),gridspec_kw={'hspace': 0.3,'wspace': 0.05})

ax_v = ax[0,0]



fig_,ax_v = dps.plot_age_moments_vertical_profile(25/2,1e3*qvcomp0* (24*60**2),1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "Spectral_r",
                                                    time_avg = True,lnP=False,custom = True, ax = ax_v,steps=51,min_val = -25/2,ext = "both",theta = True)

ax_v.set_title("$vq$ [$g \\cdot kg^{-1} \\cdot m \\cdot s^{-1}$]\n     (A)",fontsize = 30,pad = 20)
ax_v.set_xlabel(" ")
ax_str = ax[0,1]


cs=ax_str.contour(lat, pfull, streamfct0/1e7,colors="grey",levels = np.linspace(-12,12,25))
plt.clabel(cs, inline=True, fontsize=14)


cb=ax_str.contourf(lat, pfull, streamfct0/1e7,cmap = "Spectral_r",levels = np.linspace(-10,10,51),extend = "both")
cbar = plt.colorbar(cb,shrink=0.8,orientation='vertical',pad=0.01)
cbar.ax.tick_params(labelsize=20)
ax_str.set_title("$\\psi$ [$10^{4} g\\cdot s^{-1}$]\n    (B)",fontsize = 30,pad = 20)





ax_vp = ax[1,0]

fig_,ax_vp = dps.plot_age_moments_vertical_profile(12.5,1e3*vqprime0* (24*60**2),1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "Spectral_r",
                                                    time_avg = True,lnP=False,custom = True, ax = ax_vp,steps=51,min_val = -12.5,ext = "neither")

ax_vp.set_title("$v'q'$ [$g \\cdot kg^{-1} m s^{-1}$]\n (C)",fontsize = 30,pad = 20)

ax_ed = ax[1,1]
fig_,ax_ed = dps.plot_age_moments_vertical_profile(100,0.1*eddykinprime0* (24*60**2),1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "Spectral_r",
                                                    time_avg = False,lnP=False,custom = True, ax = ax_ed,steps=51,min_val = -100,ext = "both")

ax_ed.set_title("$v'K'$ [$10\\cdot m^3 s^{-3}$]\n (D)",fontsize = 30,pad = 20)
ax_ed.set_xlabel(" ")
ax_ed.set_ylabel(" ")




for i in range(2):
    for j in range(2):
        ax[i,j].set_ylim(975,200)
        ax[1,j].set_xlabel("Latitude [deg]", fontsize = 24)
        ax[i,0].set_ylabel("Pressure [hPa]", fontsize = 24)
        ax[i,j].tick_params(axis='x', labelsize=18)  # Change font size for x-axis tick labels
        ax[i,j].tick_params(axis='y', labelsize=18)
        
plt.savefig(f"{path_cur}/circulation_vert.png",dpi = 250,bbox_inches = "tight")
plt.close()
print("Saved circulation_vert.png",file=sys.stdout, flush=True)
# -------------------------------------------------------------------
fig, ax = plt.subplots(2,2,figsize = (22,16),gridspec_kw={'hspace': 0.3,'wspace': 0.05})

ax_v = ax[0,0]



fig_,ax_v = dps.plot_age_moments_vertical_profile(1.5,1e3*(qvcomp2 - qvcomp0)* (24*60**2)/delta_sst_exact,1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "BrBG",
                                                    time_avg = True,lnP=False,custom = True, ax = ax_v,steps=51,min_val = -1.5,ext = "both")

ax_v.set_title("$vq$ [$g \\cdot kg^{-1}  m  s^{-1} K^{-1}$]\n     (A)",fontsize = 30,pad = 20)
ax_v.set_xlabel(" ")
ax_str = ax[0,1]


cs=ax_str.contour(lat, pfull, (streamfct0)/1e6,colors="grey",levels = np.linspace(-120,120,25))
plt.clabel(cs, inline=True, fontsize=14)


cb=ax_str.contourf(lat, pfull, (streamfct2-streamfct0)/1e10/delta_sst_exact,cmap = "BrBG",levels = np.linspace(-1,1,51),extend = "both")
plt.colorbar(cb,shrink=0.8,orientation='vertical',pad=0.01)

ax_str.set_title("$\\psi$ [$10^{10} kg s^{-1} K^{-1}$]\n    (B)",fontsize = 30,pad = 20)





ax_vp = ax[1,0]

fig_,ax_vp = dps.plot_age_moments_vertical_profile(1.5,1e3*(vqprime2 - vqprime0)* (24*60**2)/delta_sst_exact,1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "BrBG",
                                                    time_avg = True,lnP=False,custom = True, ax = ax_vp,steps=51,min_val = -1.5,ext = "both")

ax_vp.set_title("$v'q'$ [$g \\cdot kg^{-1} \\cdot m \\cdot s^{-1}\\cdot K^{-1}$]\n (C)",fontsize = 30,pad = 20)

ax_ed = ax[1,1]
fig_,ax_ed = dps.plot_age_moments_vertical_profile(40,0.1*(eddykinprime2-eddykinprime0)* (24*60**2)/delta_sst_exact,1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "BrBG",
                                                    time_avg = True,lnP=False,custom = True, ax = ax_ed,steps=51,min_val = -40,ext = "both")

ax_ed.set_title("$v'K'$ [$10\\cdot m^3 s^{-3}$]\n (D)",fontsize = 30,pad = 20)
ax_ed.set_xlabel(" ")
ax_ed.set_ylabel(" ")


for i in range(2):
    for j in range(2):
        ax[i,j].set_ylim(975,200)
        ax[1,j].set_xlabel("Latitude [deg]", fontsize = 24)
        ax[i,0].set_ylabel("Pressure [hPa]", fontsize = 24)
        ax[i,j].tick_params(axis='x', labelsize=18)  # Change font size for x-axis tick labels
        ax[i,j].tick_params(axis='y', labelsize=18)
fig.suptitle("Anomally",fontsize = 35)    
plt.savefig(f"{path_cur}/circulation_vert_anomaly.png",dpi = 250,bbox_inches = "tight")
plt.close()
print("Saved circulation_vert_anomaly.png",file=sys.stdout, flush=True)
# -------------------------------------------------------------------
qomega0 = omega0 * q0
qomega2 = omega2 * q2
vert_qomega0 = dps.vertical_int(qomega0.values,phalf,psint)
vert_qomega2 = dps.vertical_int(qomega2.values,phalf,psint)

P0 = ds_dict['0']['ds_age'].precipitation
P2 = ds_dict['2']['ds_age'].precipitation
# -------------------------------------------------------------------

fig, ax = plt.subplots(nrows=1, ncols =2, 
                            subplot_kw={'projection': ccrs.Robinson()},figsize = (22,6),gridspec_kw={'hspace': 0.01,'wspace': 0.01})
ax_age_vert = ax[0]

dps.plot_2d(-np.mean(vert_qomega0,axis = 0),lat,lon,ax_age_vert,"Spectral_r",level_space=np.linspace(-2,2,51),land_bool=True,land_mask=land_mask,proj = ccrs.PlateCarree(),extend = "both")
ax_age_vert.set_title(" ")
ax_age_vert.set_xlabel("lon [deg]")
ax_age_vert.set_ylabel("lat [deg]")
# Add gridlines
gl = ax_age_vert.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False
ax_age_vert.set_title("$-\\{\\omega \\cdot q\\}$ [$g\\cdot kg \\cdot m^{-3}s^{-3}$]\n (A)",pad = 20,fontsize = 30)

ax_age_vert = ax[1]

dps.plot_2d(1e3*np.mean(P0.values,axis = 0),lat,lon,ax_age_vert,"YlGnBu",level_space=np.linspace(0,0.25,51),land_bool=True,land_mask=land_mask,proj = ccrs.PlateCarree(),extend = "neither")
ax_age_vert.set_title(" ")
ax_age_vert.set_xlabel("lon [deg]")
ax_age_vert.set_ylabel("lat [deg]")
# Add gridlines
gl = ax_age_vert.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False
ax_age_vert.set_title("P [$g\\cdot m^{-2} s^{-1}$]\n (B)",pad = 20,fontsize = 30)


plt.savefig(f"{path_cur}/circ2D.png",dpi = 250,bbox_inches = "tight")
plt.close()
print("Saved circ2D.png",file=sys.stdout, flush=True)
# -------------------------------------------------------------------

fig, ax = plt.subplots(nrows=1, ncols =2, 
                            subplot_kw={'projection': ccrs.Robinson()},figsize = (22,7),gridspec_kw={'hspace': 0.01,'wspace': 0.01})
ax_age_vert = ax[0]

dps.plot_2d(np.mean((-vert_qomega2 + vert_qomega0),axis = 0)/delta_sst_exact,lat,lon,ax_age_vert,"BrBG",level_space=np.linspace(-1,1,51),land_bool=True,land_mask=land_mask,proj = ccrs.PlateCarree(),extend = "both")


# Add gridlines
gl = ax_age_vert.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False
ax_age_vert.set_title("$-\\{\\omega \\cdot q\\}$ [$g\\cdot kg \\cdot m^{-3}s^{-3} K^{-1}$]\n (A)",pad = 20,fontsize = 30)

ax_age_vert = ax[1]

dps.plot_2d(1e3*np.mean((P2-P0).values,axis = 0)/delta_sst_exact,lat,lon,ax_age_vert,"BrBG",level_space=np.linspace(-0.05,0.05,51),land_bool=True,land_mask=land_mask,proj = ccrs.PlateCarree(),extend = "both")


# Add gridlines
gl = ax_age_vert.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False
ax_age_vert.set_title("P [$g\\cdot m^{-2} s^{-1} K^{-1}$]\n (B)",pad = 20,fontsize = 30)

fig.suptitle("Anomaly",fontsize = 35)

plt.savefig(f"{path_cur}/circ2D_anomaly.png",dpi = 250,bbox_inches = "tight")
plt.close()
print("Saved circ2D_anomaly.png",file=sys.stdout, flush=True)
# -------------------------------------------------------------------



fig, ax = plt.subplots(2,2,figsize = (22,16),gridspec_kw={'hspace': 0.2,'wspace': 0.05})
ax_age = ax[0,0]

fig_,ax_age = dps.plot_age_moments_vertical_profile(20,mean,1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "YlGnBu",
                                                    time_avg = True,lnP=False,custom = True, col = "white",ax = ax_age,steps=41)
ax_age.set_ylim(975,200)

pinds = [925,700,400,925,700,400,925,700,400]
latinds = [0,0,0,55,55,55,82,82,82]

ax_age.scatter(latinds, pinds, marker = "x", color = "black",s=100)


ax_age_strat = ax[1,0]
fig_,ax_age_strat = dps.plot_age_moments_vertical_profile(1500,mean,1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "YlGnBu",
                                                    time_avg = True,lnP=True,custom = True, ax = ax_age_strat,steps = 31)


ax_age_strat.set_ylim(200,5)

ax_shape = ax[0,1]

fig_,ax_shape = dps.plot_age_moments_vertical_profile(2,(24*60**2)*shape,1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "coolwarm",
                                                    time_avg = True,lnP=False,custom = True, ax = ax_shape,steps =41)

ax_shape.scatter(latinds, pinds, marker = "x", color = "black",s=100)
ax_shape.set_ylim(975,200)
ax_shape_strat = ax[1,1]
fig_,ax_shape_strat = dps.plot_age_moments_vertical_profile(2,(24*60**2)*shape,1,ps,pfull,lat,title = " ",
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "coolwarm",
                                                    time_avg = True,lnP=True,custom = True, ax = ax_shape_strat,col = "black",steps = 41)
ax_shape_strat.set_ylim(200,5)

plt.subplots_adjust(left=0.17) 
fig.text(0.11, 0.7, 'Troposphere', va='center', ha='center', rotation='vertical', fontsize=30)
fig.text(0.11, 0.28, 'Stratosphere', va='center', ha='center', rotation='vertical', fontsize=30)



letters = ["Mean WVRT [Days]\n (A)","(C)","Shape Parameter\n (B)","(D)"]
k=0
for i in range(2):
    for j in range(2):
        ax[j,i].set_title(letters[k],fontsize = 30,pad = 20)
        ax[j,i].plot(lat,trop_h_0,color = "hotpink",linewidth = 3,linestyle = "dashed")
        k+=1



ax_age.set_xlabel(" ")
ax_shape.set_xlabel(" ")

ax_shape.set_ylabel(" ")
ax_shape_strat.set_ylabel(" ")
fig.suptitle("Control Experiment",fontsize = 35)
plt.savefig(f"{path_cur}/vert_profile_age_shape.png",dpi = 250,bbox_inches = "tight")
plt.close()
print("Saved vert_profile_age_shape.png",file=sys.stdout, flush=True)
# -------------------------------------------------------------------

fig, ax = plt.subplots(2,2,figsize = (22,16),gridspec_kw={'hspace': 0.2,'wspace': 0.05})
ax_age = ax[0,0]
title_age = "Zonal Average of mean Water Vapour Residence Time"
fig_,ax_age = dps.plot_age_moments_vertical_profile(15,(24*60**2)*rel_anomaly_mean,1,ps,pfull,lat,title = title_age,
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "BrBG",
                                                    time_avg = True,lnP=False,custom = True, ax = ax_age,min_val = -15,steps = 51,ext = "both")
ax_age.set_ylim(975,200)


ax_age_strat = ax[1,0]
fig_,ax_age_strat = dps.plot_age_moments_vertical_profile(15,(24*60**2)*rel_anomaly_mean,1,ps,pfull,lat,title = title_age,
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "BrBG",
                                                    time_avg = True,lnP=True,custom = True, ax = ax_age_strat,col = "black",min_val = -15,steps = 51,ext = "both")


ax_age_strat.set_ylim(200,8)
ax_shape = ax[0,1]
title_shape = "Zonal Average of Shape parameter"
fig_,ax_shape = dps.plot_age_moments_vertical_profile(25,(24*60**2)*rel_anomaly_shape,1,ps,pfull,lat,title = title_shape,
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "PuOr",
                                                    time_avg = True,lnP=False,custom = True, ax = ax_shape,min_val = -25,steps = 51,ext = "both")
ax_shape.set_ylim(975,200)
ax_shape_strat = ax[1,1]
fig_,ax_shape_strat = dps.plot_age_moments_vertical_profile(25,(24*60**2)*rel_anomaly_shape,1,ps,pfull,lat,title = title_shape,
                                                    pot_temp = pot_temp_zonal,central=True,cmap = "PuOr",
                                                    time_avg = True,lnP=True,custom = True, ax = ax_shape_strat,col = "black",min_val = -25,steps = 51,ext = "both")

ax_shape_strat.set_ylim(200,8)
plt.subplots_adjust(left=0.17) 
fig.text(0.11, 0.7, 'Troposphere', va='center', ha='center', rotation='vertical', fontsize=30)
fig.text(0.11, 0.28, 'Stratosphere', va='center', ha='center', rotation='vertical', fontsize=30)


letters = ["Mean WVRT [Days]\n (A)","(C)","Shape Parameter\n (B)","(D)"]
k=0
for i in range(2):
    for j in range(2):
        ax[j,i].set_title(letters[k],fontsize = 25,pad = 12)
        ax[j,i].plot(lat,trop_h_0,color = "dodgerblue",linewidth = 3,linestyle = "dashed",label = r"[control]")
        ax[j,i].plot(lat,trop_h_2,color = "tomato",linewidth = 3,linestyle = "dashed",label = r"[+2K]")
        k+=1
ax[0,0].legend(fontsize = 20)      
        
ax_age.set_xlabel(" ")
ax_shape.set_xlabel(" ")
ax_shape.set_ylabel(" ")
ax_shape_strat.set_ylabel(" ")
fig.suptitle(r"Relative anomaly [$\% K^{-1}$]",fontsize = 35,usetex = True)
plt.savefig(f"{path_cur}/vert_profile_age_shape_rel_anomaly.png",dpi = 250,bbox_inches = "tight")
plt.close()
print("Saved vert_profile_age_shape_rel_anomaly.png",file=sys.stdout, flush=True)
# -------------------------------------------------------------------
v_age_time = ds_dict["0"]["vert_time_mean"]
v_shape_time = ds_dict["0"]["vert_time_shape"]
v_std_time = ds_dict["0"]["v_std_time"]
lat = ds_dict["0"]["lat"]
lon = ds_dict["0"]["lon"]
ps = ds_dict["0"]["ps"]
pfull = ds_dict["0"]["pfull"]


fig, ax = plt.subplots(nrows=2, ncols=2, 
                            subplot_kw={'projection': ccrs.Robinson()},figsize = (22,12),gridspec_kw={'hspace': 0.01,'wspace': 0.01})


ax_age_920 = ax[0,0]
dps.plot_age_moments_2d(920,15,mean,1,ps,lat,lon,pfull,fig, ax_age_920, time_avg = True,cmap = "YlGnBu",
                        title = "920 hPa",land_bool = True,land_mask = land_mask,steps = 31)

ax_age_920.set_title("920 hPa")
ax_age_920.set_xlabel("lon [deg]")
ax_age_920.set_ylabel("lat [deg]")
# Add gridlines
gl = ax_age_920.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False

ax_age_600 = ax[1,0]
dps.plot_age_moments_2d(600,15,mean,1,ps,lat,lon,pfull,fig, ax_age_600, time_avg = True,cmap = "YlGnBu",
                        title = "600 hPa",land_bool = True,land_mask = land_mask,steps = 31)

ax_age_600.set_title("600 hPa")
ax_age_600.set_xlabel("lon [deg]")
ax_age_600.set_ylabel("lat [deg]")
# Add gridlines
gl = ax_age_600.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False


ax_shape_920 = ax[0,1]
dps.plot_shape_param_2d(920,2,shape,ps,lat,lon,pfull,fig, ax_shape_920, time_avg = True,cmap = "coolwarm",
                        title = "920 hPa",land_bool = True,land_mask = land_mask,steps = 41)


ax_shape_920.set_xlabel("lon [deg]")
ax_shape_920.set_ylabel("lat [deg]")
# Add gridlines
gl = ax_shape_920.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False

ax_shape_600 = ax[1,1]
dps.plot_shape_param_2d(600,2,shape,ps,lat,lon,pfull,fig, ax_shape_600, time_avg = True,cmap = "coolwarm",
                        title = "600 hPa",land_bool = True,land_mask = land_mask,steps = 41)




fig.text(0.1,0.3, '600 hPa', va='center', rotation='vertical',ha='center', fontsize=30)
fig.text(0.1,0.69, '920 hPa', va='center', ha='center',rotation='vertical', fontsize=30)

letters = ["Mean WVRT [Days]\n (A)","(C)","Shape Parameter\n (B)","(D)"]
k=0
for i in range(2):
    for j in range(2):
        ax[j,i].set_title(letters[k],pad = 20,fontsize = 30)
        k+=1

ax_shape_600.set_xlabel("lon [deg]")
ax_shape_600.set_ylabel("lat [deg]")
# Add gridlines
gl = ax_shape_600.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False
fig.suptitle("Control Experiment",fontsize = 35)

plt.savefig(f"{path_cur}/maps_age_shape.png",dpi = 250,bbox_inches = "tight")
plt.close()
print("Saved maps_age_shape.png",file=sys.stdout, flush=True)
# -------------------------------------------------------------------
fig, ax = plt.subplots(nrows=2, ncols=2, 
                            subplot_kw={'projection': ccrs.Robinson()},figsize = (22,12),gridspec_kw={'hspace': 0.01,'wspace': 0.01})


ax_age_920 = ax[0,0]
dps.plot_age_moments_2d(920,15,24*60**2*rel_anomaly_mean,1,ps,lat,lon,pfull,fig, ax_age_920, time_avg = True,cmap = "BrBG",
                        title = "920 hPa",land_bool = True,land_mask = land_mask,min_val=-15,steps = 51,ext = "both")


ax_age_920.set_xlabel("lon [deg]")
ax_age_920.set_ylabel("lat [deg]")
# Add gridlines
gl = ax_age_920.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False

ax_age_600 = ax[1,0]
dps.plot_age_moments_2d(600,15,24*60**2*rel_anomaly_mean,1,ps,lat,lon,pfull,fig, ax_age_600, time_avg = True,cmap = "BrBG",
                        title = "600 hPa",land_bool = True,land_mask = land_mask,min_val=-15,steps=51,ext = "both")


ax_age_600.set_xlabel("lon [deg]")
ax_age_600.set_ylabel("lat [deg]")
# Add gridlines
gl = ax_age_600.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False




ax_shape_920 = ax[0,1]
dps.plot_shape_param_2d(920,10,rel_anomaly_shape,ps,lat,lon,pfull,fig, ax_shape_920, time_avg = True,cmap = "PuOr",
                        title = "920 hPa",land_bool = True,land_mask = land_mask,min_val=-10,steps = 51,ext = "both")


ax_shape_920.set_xlabel("lon [deg]")
ax_shape_920.set_ylabel("lat [deg]")
# Add gridlines
gl = ax_shape_920.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False

ax_shape_600 = ax[1,1]
dps.plot_shape_param_2d(600,10,rel_anomaly_shape,ps,lat,lon,pfull,fig, ax_shape_600, time_avg = True,cmap = "PuOr",
                        title = "600 hPa",land_bool = True,land_mask = land_mask,min_val=-10,steps = 51,ext = "both")

ax_shape_600.set_xlabel("lon [deg]")
ax_shape_600.set_ylabel("lat [deg]")


fig.text(0.1,0.3, '600 hPa', va='center', rotation='vertical',ha='center', fontsize=30)
fig.text(0.1,0.69, '920 hPa', va='center', ha='center',rotation='vertical', fontsize=30)


k=0
for i in range(2):
    for j in range(2):
        ax[j,i].set_title(letters[k],pad = 20,fontsize = 25)
        k+=1

# Add gridlines
gl = ax_shape_600.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False

fig.suptitle(r"Relative anomaly [$\% K^{-1}$]",fontsize = 35)
plt.savefig(f"{path_cur}/maps_age_shape_rel_anomaly.png",dpi = 250,bbox_inches = "tight")
plt.close()
print("Saved maps_age_shape_rel_anomaly.png",file=sys.stdout, flush=True)
# -------------------------------------------------------------------
precip_age = ds_dict["0"]["precip_age"]
anomaly_precip_age = 100*((ds_dict["2"]["precip_age"] - ds_dict["0"]["precip_age"])/ ds_dict["0"]["precip_age"])/delta_sst_exact

fig, ax = plt.subplots(nrows=1, ncols=2, 
                            subplot_kw={'projection': ccrs.Robinson()},figsize = (22,7),gridspec_kw={'hspace': 0.01,'wspace': 0.01})

ax_age_vert = ax[0]
dps.plot_2d(precip_age.values/(24*60**2),lat,lon,ax_age_vert,"YlGnBu",level_space=np.linspace(0,15,31),land_bool=True,land_mask=land_mask,proj = ccrs.PlateCarree())
ax_age_vert.set_title("Control Experiment",fontsize = 25,pad = 10)

# Add gridlines
gl = ax_age_vert.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False
ax_age_vert.set_title("$\\nu$ [Days]\n (A)",fontsize = 30,pad = 20)
ax_age_vert = ax[1]
dps.plot_2d(ds_dict['0']["vert_time_mean"].values,lat,lon,ax_age_vert,"YlGnBu",level_space=np.linspace(0,15,31),land_bool=True,land_mask=land_mask,proj = ccrs.PlateCarree())
ax_age_vert.set_title("$\\{q\\tau\\}/\\{q\\}$ [Days]\n (B)",fontsize = 30,pad = 20)
ax_age_vert.set_xlabel("lon [deg]")
ax_age_vert.set_ylabel("lat [deg]")
# Add gridlines
gl = ax_age_vert.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False

fig.suptitle("Control Experiment",fontsize = 35)
plt.savefig(f"{path_cur}/map_precip_age.png",dpi = 250,bbox_inches = "tight")
plt.close()
print("Saved map_precip_age.png",file=sys.stdout, flush=True)
# -------------------------------------------------------------------

fig, ax = plt.subplots(nrows=1, ncols=2, 
                            subplot_kw={'projection': ccrs.Robinson()},figsize = (22,6),gridspec_kw={'hspace': 0.01,'wspace': 0.01})

ax_age_vert = ax[0]
dps.plot_2d(anomaly_precip_age.values,lat,lon,ax_age_vert,"BrBG",level_space=np.linspace(-30,30,31),land_bool=True,land_mask=land_mask,proj = ccrs.PlateCarree(),extend = "both")
#ax_age_vert.set_title("Control Experiment",fontsize = 25,pad = 10)

# Add gridlines
gl = ax_age_vert.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False
ax_age_vert.set_title("$\\nu$\n (A)",fontsize = 30,pad = 20)
ax_age_vert = ax[1]
dps.plot_2d(rel_anomaly_vert_mean.values,lat,lon,ax_age_vert,"BrBG",level_space=np.linspace(-30,30,31),land_bool=True,land_mask=land_mask,proj = ccrs.PlateCarree(),extend = "neither")
ax_age_vert.set_title("$\\{q\\tau\\}/\\{q\\}$\n (B)",fontsize = 30,pad = 20)
ax_age_vert.set_xlabel("lon [deg]")
ax_age_vert.set_ylabel("lat [deg]")
# Add gridlines
gl = ax_age_vert.gridlines(draw_labels=True, linestyle='--', color='gray')
gl.top_labels = False
gl.right_labels = False

fig.suptitle(r"Relative anomaly [$\% K^{-1}$]",fontsize = 35)

plt.savefig(f"{path_cur}/map_precip_age_anomaly.png",dpi = 250,bbox_inches = "tight")
plt.close()
print("Saved map_precip_age_anomaly.png",file=sys.stdout, flush=True)
print("Done!",file=sys.stdout, flush=True)
