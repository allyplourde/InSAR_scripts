import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import griddata
from PIL import Image

from geocode import coord_to_sarpix
from geocode import infer_elevations
from sarlab.met import parse_ec_dir
from sarlab.gammax import readBin, writeBin, SLC_Par, MLI_Par


EC_DIR_IN = '/local-scratch/users/aplourde/met_data/env_canada/Inuvik/'
EC_DIR_TV = '/local-scratch/users/aplourde/met_data/env_canada/TrailValley/'
ERA5_DIR = '/local-scratch/users/aplourde/met_data/era5/'
#sar_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/'
#sar_dir = '/local-scratch/users/jaysone/projects_active/inuvik/RS2_SLA27_D/'
#sub_dir = 'full_scene/'
sar_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/TSX_SM39_D/'
sub_dir = 'full_scene_crop/'
#sub_dir = 'crop_sites/'
refdem = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/dems/tdx/topo_indices_new/dem_cropped_new.tif'
#refdem = '/local-scratch/users/jaysone/projects_active/inuvik/RS2_SLA27_D/full_scene/refdem/dims_op_oc_dfd2_641407837_5/TDM.DEM.DEM/TDM1_DEM__04_N68W134_V01_C/DEM/TDM1_DEM__04_N68W134_DEM.tif'


### https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation ###
#PARAM = 'snow_density'; ext = '.rsn'
PARAM = 'snow_depth'; ext = '.sde'; units = 'm'
#PARAM = 'snow_depth_water_equivalent'; ext = '.swe'
#PARAM = '10m_u_component_of_wind'; ext = '.uwnd'

#start_date = pd.to_datetime('2019-06-01')
#end_date = pd.to_datetime('2022-10-01')

def getECMetData(df, met_station = None, met_dir = None):
    """ Gets Environment Canada Meteorological data for corresponding
    DateTimeIndex from input dataframe and returns the dataframe with
    the parsed meterological data 
    """
    if met_station == "Inuvik":
        met_dir = EC_DIR_IN
    elif met_station == "TrailValley":
        met_dir = EC_DIR_TV
    elif met_dir is None:
        print("Error! Enter valid station name or valid EC directory.")
        return
    dict = parse_ec_dir(met_dir, freq='daily', plot=False)

    met_df = pd.DataFrame(data=dict)

    met_df.index = pd.to_datetime(met_df['Date/Time'])

    out_df = df.merge(met_df, how='left', left_index=True, right_index=True)
    #print(out_df['Snow on Grnd (cm)'])
    return out_df


def getERA5MetData(dates, grib_file, look_str='1_1', met_dir=ERA5_DIR):
    """ Gets ERA5-Reanalysis Meteorological data for corresponding
    dates abd create 2D maps in the sarpix geometry
    """
    print(met_dir + grib_file)
    ds = xr.open_dataset(met_dir + grib_file, engine="cfgrib", backend_kwargs={'filter_by_keys': {'edition': 2}})
    print(ds)
    var = list(ds)[0]

    # get coordinates
    lngs = ds.longitude.values - 360 
    lats = ds.latitude.values
    lngs, lats = np.meshgrid(lngs, lats)
    grid_shape = lngs.shape
    lngs = lngs.flatten()
    lats = lats.flatten()

    #demfile = sar_dir + sub_dir + "refdem_tdx/dims_op_oc_dfd2_641407837_5/TDM.DEM.DEM/TDM1_DEM__04_N68W134_V01_C/DEM/TDM1_DEM__04_N68W134_DEM.tif"
    hgts = infer_elevations(refdem, lngs, lats)

    # convert to sar pixel geometry
    coord_file = met_dir + grib_file.split('.')[0] + '.coords'
    #SLC_par = sar_dir + sub_dir +'rslc/master.rslc.par'
    MLI_par = sar_dir + sub_dir + 'rmli_'+look_str+'/rmli_'+look_str+'.ave.par'
    sarpix = coord_to_sarpix(lats, lngs, hgts, coord_file, MLI_par, sar_dir=sar_dir+sub_dir, OFF_par=None, looks=look_str)

    # get subset corresponding to InSAR stack
    ds_times = pd.to_datetime(ds.time.values)
    dates = np.array(dates)
    sarvar = ds[var].values[np.isin(ds_times, dates)]
    
    # create array (times, data)
    out = []
    for (t, v) in zip(dates, sarvar):
        out.append([t, v])

    out = np.array(out, dtype=object)

    return out, sarpix


def mk_era5_maps(ifg_dates, era5_data, era5_coord_map, look_str, show_results=False):

    master_par = MLI_Par(sar_dir + sub_dir + 'rmli_' + look_str + '/rmli_' + look_str + '.ave.par')
    dim = master_par.dim

    ave_rmli = sar_dir + sub_dir + 'rmli_' + look_str + '/rmli_' + look_str + '.ave'

    sar_im = readBin(ave_rmli, dim, 'float32')

    r_px = era5_coord_map.T[0]
    az_px = era5_coord_map.T[1]
    era5_grid = era5_coord_map[:,:2]

    # create coordinate grid on which to interpolate era5 data
    r_coords = range(int(np.max(r_px)) + 5)
    az_coords = range(int(np.max(az_px)) + 5)
    r_grid, az_grid = np.meshgrid(r_coords, az_coords)
    sar_grid = np.column_stack([r_grid.ravel(), az_grid.ravel()])

    for ifg in ifg_dates:
        mdate = ifg[0]
        sdate = ifg[1]

        figname = mdate.strftime('%Y%m%d') + '_' + sdate.strftime('%Y%m%d') + '.png'
        imname = mdate.strftime('%Y%m%d') + '_' + sdate.strftime('%Y%m%d') + ext


        m_idx = np.where(era5_data[:, 0] == mdate)[0][0]
        s_idx = np.where(era5_data[:, 0] == sdate)[0][0]

        delta_era5 = era5_data[s_idx][1] - era5_data[m_idx][1]
        print(f"{era5_data[s_idx][1]} - {era5_data[m_idx][1]} = {delta_era5}")

        plt.figure(figsize=(18, 6))

        plt_max = np.max(delta_era5)
        plt_min = np.min(delta_era5)

        title = mdate.strftime('%Y%m%d') + ' - ' + sdate.strftime('%Y%m%d')
        plt.suptitle(title)

        plt.subplot(131)
        plt.title("ERA5")
        plt.imshow(delta_era5, cmap='coolwarm', vmin=plt_min, vmax=plt_max)
        plt.colorbar()

        plt.subplot(132)
        plt.title("Average Intensity")
        plt.imshow(20*np.log10(sar_im.T), vmin=-50)
        plt.scatter(r_px, az_px, c=delta_era5, cmap='coolwarm', vmin=plt_min, vmax=plt_max)
        plt.colorbar()

        # Interpolate the values from the era5 grid onto the sar grid
        interpolated_values = griddata(era5_grid, delta_era5.flatten(), sar_grid)

        # Reshape the interpolated values into a 2D array
        interpolated_values = interpolated_values.reshape(r_grid.shape)

        plt.subplot(133)
        plt.title('ERA5 Interpolated onto SAR geometry')
        plt.imshow(interpolated_values[:sar_im.shape[1], :sar_im.shape[0]], cmap='coolwarm', vmin=plt_min, vmax=plt_max)
        print(interpolated_values.shape, sar_im.shape)
        plt.colorbar()

        out_dir = ERA5_DIR+'delta_' + PARAM + '/' + sar_dir.split('/')[-2] + '/' + sub_dir + 'swe_' + look_str + '/'

        plt.savefig(out_dir + figname)
        if show_results:
            plt.show()
        plt.close()

        sde = interpolated_values[:sar_im.shape[1], :sar_im.shape[0]].T
        writeBin(out_dir + imname, sde)

        #im = readBin(ERA5_DIR + 'delta_snow_depth/' + imname, dim, 'float32')
        #plt.imshow(im.T)
        #plt.show()


def crop_era5_maps(maps, dim=[7621,11393], crop=[4500, 3492, 996, 2592]):
    for map in maps:

        try:
            im = readBin(map, dim, 'float32')
            print(f"succefully read {map}")
            crop_im = im[crop[0]:crop[0] + crop[2], crop[1]:crop[1] + crop[3]]
            dirname, filename = os.path.split(map)

            cropdir = dirname + '/crop_sites/'
            if not os.path.exists(cropdir):
                os.makedirs(cropdir)
            writeBin(cropdir + filename, crop_im)
        except:
            #print("mismatch dimensions... skipping.")
            print(map)


def importSnowFieldData(file):

    # import data
    df = pd.read_csv(file)
    cols = list(df)

    date = pd.to_datetime(df[cols[0]].values, utc=True).tz_localize(None)

    # ensure consistent data types
    out = pd.DataFrame(data = {'date': date, 'snow_depth': np.float64(df[cols[1]])})

    return out


def processSnowFieldData(file):
    data = importSnowFieldData(file)

    out = pd.DataFrame(index = data.date)

    # retrieve logger snow depth
    out['snow_depth_cm'] = data['snow_depth'].values
    out['snow_depth_cm'] = data['snow_depth'].values * 25.4
    out['snow_depth_cm'] = data['snow_depth'].values * 2.54

    # ensure data is in order
    out = out.sort_index()

    # truncate data to start/end date
    out = out.loc[out.index >= start_date]
    out = out.loc[out.index <= end_date]

    # throwaway errouneous data (sensor may malfuction below -40C)
    #stdev = np.std(out['h1_cm'])
    #mean = np.mean(out['h1_cm'])
    #out[out['h1_cm'] < mean - 4*stdev] = np.nan
    #out[out['h1_cm'] > mean + 4*stdev] = np.nan
    #stdev = np.std(out['h2_cm'])
    #mean = np.mean(out['h2_cm'])
    #out[out['h2_cm'] < mean - 4*stdev] = np.nan
    #out[out['h2_cm'] > mean + 4*stdev] = np.nan

    return out


def processFieldwithEC():
    data_dir = '/local-scratch/users/aplourde/field_data/'
    files = glob.glob(data_dir + '/*/*snow_depth.csv')
    snow_depth = {}
    for file in files:
        site = re.search(r'site_.', file).group(0)
        data = processSnowFieldData(file)

        snow_depth[site] = data.resample('D').mean()
        snow_depth[site] = getECMetData(snow_depth[site], met_station="Inuvik")

        # print( snow_depth[site])
        # print(list(snow_depth[site]))
        snow_depth[site].to_csv(file[-4] + '_processed.csv')


def getSARdates(itab, stab):
    itab = np.loadtxt(itab, dtype='int')
    stab = np.loadtxt(stab, dtype='str')

    idates = []
    for i in itab:
        m = os.path.basename(stab[i[0]-1][0]).split('.')[0]
        s = os.path.basename(stab[i[1]-1][0]).split('.')[0]
        if 'TSX' in sar_dir:
            m = m.split('_HH')[0]
            s = s.split('_HH')[0]
        idates.append(pd.to_datetime([m, s]))

    dates = np.unique(np.array(idates).flatten())

    return dates, idates


def snowDepthtoSWE(sde, dim = None, snow_density=0.3):
    if dim is not None:
        for file in sde:
            try:
                sde = readBin(file, dim, 'float32')
                swe = sde * snow_density
                writeBin(file.split('.')[0] + '.swe', swe)
                print(f"{file} converted to swe.")
            except:
                print("wrong image dimensions... skipping")
                continue
    else:
        return sde * snow_density


def dailyERAatSite(grib_file=PARAM + '_full_scene_2014-2023.grib', met_dir = ERA5_DIR, show_plot=False):


    lat = 68.6
    lon = -133.7+360

    print(met_dir + grib_file)
    ds = xr.open_dataset(met_dir + grib_file, engine="cfgrib", backend_kwargs={'filter_by_keys': {'edition': 2}})
    print(ds)
    var = list(ds)[0]

    site = ds.sel(latitude=lat, longitude=lon, method='nearest')

    vals = site[var].values
    dates = pd.to_datetime(site.time.values)

    if show_plot:
        plt.plot(dates, vals)
        plt.show()

    return pd.DataFrame(index=dates, data={'snow_depth_m': vals})

if __name__ == "__main__":

    looks = '2_2'

    itab = sar_dir + sub_dir + 'itab_lf_snow'
    RSLC_tab = sar_dir + sub_dir + 'RSLC_tab'

    dates, idates = getSARdates(itab, RSLC_tab)

    era5_snow_depth, era5_coord_map = getERA5MetData(dates, PARAM + '_inuvik_2012-2023.grib', look_str=looks)
    mk_era5_maps(idates, era5_snow_depth, era5_coord_map, looks, show_results=0)

    """
    file = ERA5_DIR + 'delta_' + PARAM + '/' + '20141128_20141222.sde'
    im = readBin(file, master_par.dim, 'float32')
    plt.subplot(121)
    plt.imshow(im.T)
    plt.colorbar()
    plt.subplot(122)
    looks = '2_3'
    new_dim = MLI_Par(sar_dir + sub_dir + 'rmli_' + looks + '/rmli_' + looks + '.ave.par')
    image = Image.fromarray(im)
    resampled_image = image.resize(new_dim.dim, resample=Image.BILINEAR)
    resampled_image = np.asarray(resampled_image).astype('float').T
    plt.imshow(resampled_image.T)
    plt.colorbar()
    plt.show()
    """

    #maps = glob.glob(ERA5_DIR + 'delta_snow_depth/*.sde')
    #3crop_era5_maps(maps, dim=[12507, 12716], crop=[1138, 12700, 1235, 2250])

    #crop_sites = glob.glob(ERA5_DIR + 'delta_snow_depth/crop_sites/*.sde')
    #snowDepthtoSWE(crop_sites, [996, 2592])
    #snowDepthtoSWE(crop_sites, [1235, 2250])


    #dailyERAatSite(grib_file='test.grib')
