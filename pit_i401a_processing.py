import sarlab.pygamma.pygamma_setup
import py_gamma as pg
from sarlab.pygamma import RcmStripmapStack
from sarlab.gammax import readBin
import os
import glob
import numpy as np
import gdal
import sarlab.gammax as gx
import timeseries

working_dir = '/local-scratch/users/aplourde/RCM/'
#sub_dir = 'Pit_I401A/arcticdem/december/; '#master = '20211122_hh'
sub_dir = 'Pit_I401A/summer/'; master = '20220716_hh'
#sub_dir = 'Pit_I401A/arcticdem/'; master = '20220809_hh'
#sub_dir = 'Pit_I401A/swath1/'

looks = (1, 1)
pol = 'hh'

print('Started processing')
processing_dir = working_dir + sub_dir
srtm_30_dir = '/local-scratch/common/dems/srtm_30m/data_files'
bounding_coords = None  # Unused at the moment
srtm30_tiles = [os.path.join(srtm_30_dir, 'n50_w122_1arc_v3.tif')]
stack = RcmStripmapStack(processing_dir, master, pol, looks,
                         fix_orbits=False, debug=False)

def main():

    stack.ingest()
    stack.register()
    stack.rmli()
    #stack.srtm_30m_refdem(srtm30_tiles, bounding_coords)

    bounding_coords = (68.26830, -133.96366, 68.53116, -133.18906)
    stack.arctic_2m_refdem(bounding_coords)
    stack.rdc_dem()
    stack.mk_diff(baseline_corr=True, itab_type = 'all')

    stack.averageRMLI()

    # print('Optimal looks: ')
    # print(gx.getOptimalLooks(os.path.join(processing_dir, 'slc',
    #                                       master + '.slc.par')))

def ave_snowcompact_ccd():
    im_list = '/local-scratch/users/aplourde/RCM/Pit_I401A/arcticdem/diff/snowcompactdates.txt'
    rmli_files = glob.glob('/local-scratch/users/aplourde/RCM/Pit_I401A/arcticdem/rmli/*h.rmli')

    rmli_par_name = rmli_files[0] + '.par'
    mli_par = pg.ParFile(rmli_par_name)
    rg_sz = mli_par.get_value('range_samples')
    width = rg_sz
    out = self.rmli_dir + '/ave.rmli'

    call_pg('ave_image', im_list, width, out, **self.pg_kwargs)
    call_pg('raspwr', out, width, **self.pg_kwargs)

def unwrap():
    pass


def mk_timeseries(ext):

    if 'rmli' in ext:
        dir = stack.rmli_dir
        files = glob.glob(os.path.join(dir, '*' + ext))

        master_par = os.path.join(dir, stack.master + ext + '.par')
        par = pg.ParFile(master_par)
        mli_rg = par.get_value('range_samples')
        mli_az = par.get_value('azimuth_lines')
        dim = [int(mli_rg), int(mli_az)]

        type = 'float32'

        im_list = []
        for file in files:
            if 'ave' in file:
                continue
            print(file, dim, type)
            im = readBin(file, dim, type)
            im_crop = im[1050:1200,10430:10580]
            im_list.append(im_crop)
        dim = im_list[0].shape
        timeseries.mk_avi(im_list, ext, dir, dim)


if __name__ == '__main__':

    #main()

    mk_timeseries('.rmli')
    #ave_snowcompact_ccd()

    #unwrap()
