from sarlab.gammax import *
from ap_utils import phasecmap
import os
import subprocess
import rasterio
import rasterio.plot as rplt
from rasterio.warp import calculate_default_transform, reproject, Resampling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image, ImageDraw
import geopandas
import fiona
geopandas.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
import shapely
import gdal_merge as gm
import re
import shutil


# working_dir = '/local-scratch/users/aplourde/RCM/Pit_I401A/arcticdem/'; master = '20220809_hh'; lim = [[1025, 1175],[10230, 10380]]
# working_dir = '/local-scratch/users/aplourde/RCM/Pit_I401A/summer/'; master = '20220716_hh'; lim = [[1050, 1200],[10430, 10580]]

#working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/full_scene/'
working_dir = '/local-scratch/users/aplourde/HDS/projects/southern_ITH/RS2_U76_D/new_crop_sites/'
master = '20170808'

ingest_cfg = {'password': 'S1m0nF7@SER', 'polarizations': 'HH'}                                                     #2
stack = SLC_stack(dirname=working_dir, name='', reg_mask=None, master=master, looks_hr=(2,3), looks_lr=(12,18), multiprocess=True, rdcdem_refine=True, skipmode='None', ingest_cfg=ingest_cfg)

#overlays = ['/local-scratch/users/aplourde/RCM/Pit_I401A/pit_manipulation.kml',
#                    '/local-scratch/users/aplourde/RCM/Pit_I401A/pit_control.kml']
overlays = []
#overlay_annotations = ['compact site', 'control site']

#bad_files = ['20220222_hh', '20220626_hh']


def geocode_by_file(file_to_geocode):

    width = stack.master_slc_par.dim[0]

    dem_dir = working_dir + 'dem/'
    #dem_dir = working_dir + 'dem_1_1/'
    gc_map = dem_dir + 'gc_map'
    #gc_map = dem_dir + 'gc_fine'
    dem_par = DEM_Par(filename=dem_dir + 'seg.dem_par')
    if 'cc' in file_to_geocode:
        interpmode = 1
        dtype = 0
        geo_type = 2
    elif 'rmli' in file_to_geocode:
        interpmode = 1
        dtype = 0
        geo_type = 2
    elif 'diff' in file_to_geocode:
        interpmode = 1
        dtype = 1
        geo_type = 4
    elif 'ras' in file_to_geocode:
        interpmode = 0
        dtype = 2
        geo_type = 0
    run('geocode_back', file_to_geocode, width, gc_map, file_to_geocode + '.geo', dem_par.dim[0], None, interpmode, dtype)
    run('data2geotiff', dem_par, file_to_geocode + '.geo', geo_type, file_to_geocode + '.tif')


def geocode_by_suffix(dir, suffix):
    files = glob.glob(working_dir + dir + '*' + suffix)

    for file in files:
        geocode_by_file(file)


def haversine(lat1, lon1, lat2, lon2, radius=6371.0):
    """
    Calculate the distance between two points on the Earth's surface
    using the Haversine formula.

    Arguments:
    lat1 -- Latitude of the first point in degrees.
    lon1 -- Longitude of the first point in degrees.
    lat2 -- Latitude of the second point in degrees.
    lon2 -- Longitude of the second point in degrees.
    radius -- Radius of the Earth in the desired unit (default: 6371.0 kilometers).

    Returns:
    The distance between the two points in the same unit as the Earth's radius.
    """
    # Convert degrees to radians
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return distance*1000


def downsample_tif(input_file, output_file, scale_factor):
    with rasterio.open(input_file) as src:
        # Calculate the new dimensions
        new_width = int(src.width / scale_factor)
        new_height = int(src.height / scale_factor)

        # Calculate the new transform
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        # Prepare the output profile
        output_profile = src.profile
        output_profile.update(width=new_width, height=new_height, transform=transform)

        # Perform the downsampling
        with rasterio.open(output_file, 'w', **output_profile) as dst:
            for i in range(1, src.count + 1):
                src_data = src.read(i, out_shape=(new_height, new_width), resampling=Resampling.bilinear)
                dst.write(src_data, i)



def plotif(file, lim = None, show_fig = True, save_fig=False):

    tif = rasterio.open(file)
    ov = []
    for overlay in overlays:
        if 'kml' in overlay:
            ov.append(geopandas.read_file(overlay, driver='KML'))
        elif 'shp' in overlay:
            ov.append(geopandas.read_file(overlay))

    data = tif.read(1)

    fig, ax = plt.subplots()
    if 'cc' in file or 'rmli' in file:
        im_plt = rplt.show(data, transform=tif.transform, ax=ax, cmap=cm.Greys_r)
        overlay_colors = ['yellow', 'blue']
    elif 'diff.adf' in file:
        phi = np.angle(data)
        im_plt = rplt.show(phi, transform=tif.transform, ax=ax, cmap=cmap)
        overlay_colors = ['black', 'blue']
    else:
        data[data < 0] = 0
        print(data.shape)
        im_plt = rplt.show(data, transform=tif.transform, ax=ax)
        overlay_colors = ['red']

    if len(overlays)>0:
        for i in range(len(overlays)):
            ov_plt = ov[i].plot(ax=ax, facecolor='none', edgecolor=overlay_colors[i])

    if lim is not None:
        plt.xlim(lim[0])
        plt.ylim(lim[1])

    plt.title(file.split('/')[-1].split('.')[0])
    #plt.colorbar()
    if save_fig:
        plt.savefig(file + '.png')
        plt.close()
    if show_fig:
        plt.show()
        plt.close()


def plotsar(sarfile, lim=None, show_fig = True, save_fig=False):

    overlay = [x[:-4] + '.coords.sarpix' for x in overlays]

    im, cmap, overlay_colors = sar_image(sarfile)
    print(np.nanmin(im[lim[1], lim[0]]), np.nanmax(im[lim[1], lim[0]]))
    plt.imshow(im, cmap=cmap, vmin=-2.5, vmax=2.5)

    for i in range(len(overlay)):
        ov = np.loadtxt(overlay[i])
        x = [coord[0] for coord in ov]
        y = [coord[1] for coord in ov]
        plt.plot(x, y, color=overlay_colors[i])
        if overlay_annotations is not None:
            plt.annotate(overlay_annotations[i], (np.mean(x) + (np.max(x) - np.min(x)), np.mean(y)), color=overlay_colors[i])

    plt.gca().invert_yaxis()

    if lim is not None:
        plt.xlim(lim[0])
        plt.ylim(lim[1])

    plt.title(sarfile.split('/')[-1].split('.')[0])
    plt.colorbar()
    if save_fig:
        plt.savefig(sarfile + '.png')
    if show_fig:
        plt.show()
    plt.close()


def stitch_tifs(dir):
    tifs = glob.glob(dir + '*dsm.tif')

    gm.main(['', '-o', dir + 'tmp0.tif', tifs[0], tifs[1]])
    i = 0
    for tif in tifs[2:]:
        if i < (len(tifs[2:]) - 1):
            gm.main(['', '-o', dir + 'tmp' + str(i + 1) + '.tif', dir + 'tmp' + str(i) + '.tif', tif])
        else:
            gm.main(['', '-o', dir + 'merged.tif', dir + 'tmp' + str(i) + '.tif', tif])
        os.remove(dir + 'tmp' + str(i) + '.tif')
        i += 1


def crop_tif(file, bb):

    with rasterio.open(file) as src:
        # Reproject the corner coordinates to match the image CRS
        lon1, lat1 = src.meta['transform'] * (bb[1], bb[0])
        #lon2, lat2 = src.meta['transform'] * (bb[3], bb[2])

        # Calculate the window of the desired crop based on the corner coordinates
        #window = src.window(lon1, lat1, lon2, lat2)
        window = src.window(lon1, lat1, lon1+bb[3], lat1+bb[2])
        print(window.height, window.width)

        # Calculate the affine transform for the output
        transform = src.window_transform(window)
        print(transform)

        # Update the metadata for the output
        meta = src.meta.copy()
        meta.update({
            'width': window.width,
            'height': window.height,
            'transform': transform
        })

        # Create the output raster file
        out = file.split('.')[0] + '_crop.tif'
        with rasterio.open(out, 'w', **meta) as dst:
            # Perform the cropping and write the output
            dst.write(src.read(window=window, out_shape=(src.count, window.height, window.width), resampling=Resampling.nearest))



def shp_to_sar():
    pass


def kml_to_sar(overlay, sarfile, looks='fr'):

    #get coord list
    ov = geopandas.read_file(overlay, driver='KML')

    geometry = ov.geometry.values[0]
    print(ov.geometry)

    if type(geometry) == shapely.geometry.polygon.Polygon:
        coords = geometry.exterior.coords.xy
        lngs = list(coords[0])
        lats = list(coords[1])
        hgts = np.zeros(len(lngs))
    elif type(geometry) == shapely.geometry.multipolygon.MultiPolygon:
        polygons = list(ov.geometry.values[0])
        coords = [np.array(pg.exterior.coords) for pg in polygons]
        lngs = [[coord[0] for coord in sub] for sub in coords][0]
        lats = [[coord[1] for coord in sub] for sub in coords][0]
        hgts = [[coord[2] for coord in sub] for sub in coords][0]
    else:
        return

    if hgts.all() == 0:
        print("--------------------------------------------------------")
        print("WARNING! Using manual altitude in coord to sarpix!!!")
        print("--------------------------------------------------------")
        hgts = [45 for x in hgts]

    coord_file_name = overlay.split('.')[0] + '.coords'
    SLC_par = stack.master_slc_par
    coord_to_sarpix(sarfile, lats, lngs, hgts, coord_file_name, SLC_par)

    
def coord_to_sarpix(lats, lngs, hgts, coord_file_name, SLC_par, sar_dir=working_dir, OFF_par=None, looks='fr'):
    
    formated_coords = np.asarray([lats, lngs, hgts]).T

    #make coord file
    #latitude   longitude   height
    np.savetxt(coord_file_name, formated_coords)
    MAP_coord = coord_file_name

    if os.path.exists(sar_dir + 'dem_' + looks):
        DEM_par = sar_dir + 'dem_' + looks + '/seg.dem_par'
    elif os.path.exists(sar_dir + 'dem'):
        DEM_par = sar_dir + 'dem/seg.dem_par'
    else:
        DEM_par = None

    #convert coords to sar pixels
    SAR_coord = coord_file_name + '.sarpix'
    run('coord_to_sarpix_list', SLC_par, OFF_par, DEM_par, MAP_coord, SAR_coord)
    #print('coord_to_sarpix_list', SLC_par, OFF_par, DEM_par, MAP_coord, SAR_coord)
    
    return np.loadtxt(SAR_coord)


def crop_by_polygon(sarfiles, crop_files):


    for cf in crop_files:

        crop = np.loadtxt(cf)
        crop_polygon = []
        for i in range(len(crop)):
            crop_polygon.append((crop[i][0], crop[i][1]))

        mean = []
        std = []
        dates = []
        for sf in sarfiles:
            if 'ave' in sf:
                continue
            im, cmap, oc = sar_image(sf)

            mask = Image.new('L', (im.T.shape), 0)
            ImageDraw.Draw(mask).polygon(crop_polygon, outline=1, fill=1)

            mask = np.asarray(mask)

            cropped_image = im * mask
            cropped_image[cropped_image == 0] = np.nan
            mean.append(np.nanmean(cropped_image))
            std.append(np.nanstd(cropped_image))
            dates.append(pd.to_datetime(re.findall("[0-9]{8}", sf)[0]))

        plt.scatter(dates, mean, label=cf.split('/')[-1].split('.')[0])
    plt.legend()
    plt.ylabel('Amplitude (dB)')
    plt.show()


def sar_image(sarfile):

    phi_cmap = phasecmap()
    dim = stack.master_slc_par.dim

    if "cc" in sarfile:
        im = readBin(sarfile, dim, 'float32')
        out = im.T
        cmap = cm.Greys_r
        overlay_colors = ['yellow', 'blue']
    elif '.rmli' in sarfile:
        im = readBin(sarfile, dim, 'float32')
        im[im == 0] = np.nan
        out = np.log10(im.T)
        cmap = cm.Greys_r
        overlay_colors = ['yellow', 'blue']
    elif 'diff.adf' in sarfile:
        im = readBin(sarfile, dim, 'complex64')
        phi = np.angle(im)
        out = phi.T
        cmap = phi_cmap
        overlay_colors = ['black', 'blue']

    return out, cmap, overlay_colors


def get_files(ext, sub_dir=None, looks = 'fr'):

    files = []

    if ext == 'rmli':
        if sub_dir is None:
            if looks == 'fr':
                sub_dir = stack._rmli_dir_fr
            elif looks == 'hr':
                sub_dir = stack._rmli_dir_hr
            elif looks == 'lr':
                sub_dir = stack._rmli_dir_lr
        files = glob.glob(stack._dir + sub_dir + '*' + ext)

    if any(x in ext for x in ['cc', 'diff']):
        if sub_dir is None:
            if looks == 'fr':
                sub_dir = stack._diff_dir_fr
            elif looks == 'hr':
                sub_dir = stack._diff_dir_hr
            elif looks == 'lr':
                sub_dir = stack._diff_dir_lr
        files = glob.glob(stack._dir + sub_dir + '*' + ext)

    return files


def infer_elevations(demfile, lngs, lats):
    tif = rasterio.open(demfile)
    
    hgts = np.zeros(lngs.shape)
    elevation = tif.read(1)
    rows, cols = tif.index(lngs, lats)

    for i, (row, col) in enumerate(zip(rows, cols)):
        try:
            elevation = tif.read(1, window=((row-100, row+100), (col-100, col+100)))[0][0]  # 10 m * 200 = average elevation over 2km x 2km grid
            hgts[i] = np.mean(elevation)
        except:
            hgts[i] = np.nan

    # create an array of indices for non-NaN values
    non_nan_idx = np.arange(len(hgts))[~np.isnan(hgts)]

    # interpolate the NaN values using the non-NaN values
    interp_vals = np.interp(np.arange(len(hgts)), non_nan_idx, hgts[~np.isnan(hgts)])
    hgts[np.isnan(hgts)] = interp_vals[np.isnan(hgts)]


    return hgts


def cropGammaDEM(dem_file, bb):


    dem_par = DEM_Par(dem_file + '_par')
    print(dem_par.dim)
    dem = readBin(dem_file, dem_par.dim, 'float32')

    new_corner_lat = bb[0]
    new_corner_lon = bb[1]

    crop_y = int(-(dem_par['corner_lat'] - new_corner_lat) / dem_par['post_lat'])
    crop_x = int(-(dem_par['corner_lon'] - new_corner_lon) / dem_par['post_lon'])

    easting = bb[3]
    southing = bb[2]

    new_height = int(-(dem_par['corner_lat'] - southing) / dem_par['post_lat'])
    new_width = int(-(dem_par['corner_lon'] - easting) / dem_par['post_lon'])

    #plt.imshow(dem.T)
    crop_dem = dem[crop_x:new_width, crop_y:new_height]
    #plt.imshow(crop_dem.T, vmin=-50)
    #plt.colorbar()
    #plt.show()

    print(crop_dem.shape)
    crop_file = dem_file.split('.')[0] + '.dem'
    #shutil.copyfile(dem_file + '_par', crop_file + '_par')
    #TODO: automatically overwrite *crop.dem_par
    crop_par = crop_file + '_par'
    #writeBin(crop_file, crop_dem)
    crop_dem = DEM(crop_file, par = DEM_Par(crop_par))
    crop_dem.ras()


def lidar_dem_mask(dem_file):

    par_file = dem_file + '_par'
    dem_par = DEM_Par(par_file)

    dem = readBin(dem_file, dem_par.dim, 'float32')

    dem_mask = dem.copy()
    dem_mask[dem_mask < 0.01] = 0
    dem_mask[dem_mask != 0] = 1
    #dem_mask[dem_mask == -9999] = 0

    plt.imshow(dem_mask.T, cmap=cm.Greys_r)
    plt.colorbar()
    plt.show()

    mask_file = dem_file + '.mask'
    writeBin(mask_file, dem_mask)
    run('rascc_bw', mask_file, None, dem_par.dim[0], None, None, None, None, None, 0, 1, None, None, None, mask_file + '.ras')


def lidar_dem_invals(dem_file):
    par_file = dem_file + '_par'
    dem_par = DEM_Par(par_file)

    dem = readBin(dem_file, dem_par.dim, 'float32')
    dem[dem <=0.01] = -9999
    cleaned = dem_file + '.cleaned'
    writeBin(cleaned, dem)
    run('rashgt', cleaned, None, dem_par.dim[0], None, None, None, None, None, None, None, None, None,
        cleaned + '.ras')


def compare_rdc_dems(dem1, dem2, dim):

    im1 = readBin(dem1, dim, 'float32')
    im2 = readBin(dem2, dim, 'float32')

    plt.subplot(311)
    plt.title('Lidar RDC Dem')
    plt.imshow(im1.T, vmin=50, vmax=150)
    plt.colorbar()
    plt.subplot(312)
    plt.title('TDX RDC Dem')
    plt.imshow(im2.T, vmin=50, vmax=150)
    plt.colorbar()
    plt.subplot(313)
    plt.title("Difference (m)")
    plt.imshow(im1.T - im2.T, cmap='RdBu', vmin=-15, vmax=15)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    pass
    ### Init ###
    #files_to_process = get_files('rmli', 'rmli/')

    ### Geocoding ###
    #geocode_by_file(file)
    #geocode_by_suffix('diff/', 'diff.adf')
    #geocode_by_suffix('', 'diff.natm.hds')
    #geocode_by_suffix('diff/', '.cc.cor')

    ### Plotting ###
    #sarfile = files_to_process[0]
    #for overlay in overlays:
    #    if 'kml' in overlay:
    #        kml_to_sar(overlay, sarfile)
    #for file in files_to_process:
        #plotsar(file, lim=lim, show_fig=False, save_fig=True)
        #plotif(file, lim=None, show_fig=False, save_fig=False)
    #crop_files = [x[:-4] + '.coords.sarpix' for x in overlays]
    #crop_by_polygon(files_to_process, crop_files)

    #lidar_tif = '/local-scratch/users/aplourde/DEMS/ITH_LiDAR/DSM/lidar_ITH_2021.tif'
    #plotif('/local-scratch/users/jaysone/projects_active/inuvik/dems/tdx/topo_indices/aspect_n_utm.tif')
    #plotif(lidar_dem)

    ### Utils ###
    #stitch_tifs('/local-scratch/users/aplourde/DEMS/ITH_LiDAR/DSM/')
    #downsampled_tif = lidar_tif.split('.')[0] + '_dsamp.tif'
    #downsample_tif(lidar_tif, downsampled_tif, 3)
    #plotif('/local-scratch/users/jaysone/projects_active/inuvik/dems/tdx/topo_indices/dem_cropped.tif')
    #plotif('/local-scratch/users/jaysone/projects_active/inuvik/dems/tdx/topo_indices/dem_cropped_gaussrad3.tif')
    #gamma_dem_file = ref_dem_dir = working_dir + 'refdem/crop/ref.dem'
    #bounding_box = [68.71502405, -133.96946111, 68.45327511, -133.27061878]
    #bounding_box = [68.80205155, -133.98562540, 68.4, -133.1]
    #cropGammaDEM(gamma_dem_file, bounding_box)

    #bounding_box = [68.66212991059548, -133.75466367848557, 68.57496516080708, -133.64324495820097] #RS2_U76
    #lidar_dem_file = '/local-scratch/users/aplourde/DEMS/ITH_LiDAR/DSM/lidar_ITH_2021_dsamp.dem'
    #cropGammaDEM(lidar_dem_file, bounding_box)
    #crop_tif('/local-scratch/users/aplourde/DEMS/ITH_LiDAR/DSM/lidar_ITH_2021.tif', bounding_box)
    bounding_box = [68.81718467953347, -133.99224866649013, 68.3844278141092, -133.19862035848803] #TSX_SM39
    crop_tif('/local-scratch/users/aplourde/HDS/projects/southern_ITH/dems/tdx/dims_op_oc_dfd2_641407837_5/TDM.DEM.DEM/TDM1_DEM__04_N68W134_V01_C/DEM/TDM1_DEM__04_N68W134_DEM.tif', bounding_box)

    #lidar_dem_mask(lidar_dem_file)
    #lidar_dem_invals(lidar_dem_file)

    #compare_rdc_dems(working_dir + 'dem_1_1/seg.dem.rdc', working_dir + '../crop_sites/dem_1_1/seg.dem.rdc', [996, 2592])



