import numpy as np
from netCDF4 import Dataset
import warnings
from scipy.io import savemat
import datetime
import pandas as pd
import pvlib
import os
import sys

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)


def _daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def _cal_SZA(times, lat, lon, alt):
    return pvlib.solarposition.get_solarposition(times, lat, lon, alt)


Mair = 28.97e-3
g = 9.80665
N_A = 6.02214076e23

# duration = [startdate, enddate) (enddate isn't included)
startdate = str(sys.argv[1])
enddate = str(sys.argv[2])

output_folder = '/discover/nobackup/asouri/PROJECTS/ECCOH/offline_ECCOH/inputs/'

start_date = datetime.date(int(startdate[0:4]), int(
    startdate[5:7]), int(startdate[8:10]))
end_date = datetime.date(int(enddate[0:4]), int(
    enddate[5:7]), int(enddate[8:10]))

# Group variables by MERRA2 GMI output file type.
DACList = ['NO2', 'ALK4', 'C2H6', 'C3H8', 'PRPE', 'O3',
           'CH4', 'CO', 'H2O2', 'ISOP', 'ACET', 'CH2O', 'MP']
MetList = ['QV', 'PL', 'T', 'H', 'aircol']
AODList = ['AODUP', 'AODDWN']
CloudList = ['CLOUD', 'TAUCLWUP', 'TAUCLIUP', 'TAUCLWDWN', 'TAUCLIDWN']
NxList = ['GMISTRATO3']
InputList = ['NO2', 'O3', 'CH4', 'CO', 'ISOP', 'ACET', 'C2H6', 'C3H8', 'PRPE', 'ALK4',
             'MP', 'H2O2', 'PL', 'QV', 'T', 'Lat', 'CLOUD', 'TAUCLWUP', 'TAUCLIUP', 'TAUCLWDWN',
             'TAUCLIDWN', 'ALBUV', 'GMISTRATO3', 'AODDWN', 'AODUP', 'CH2O', 'SZA','aircol']
InputAsIs = ['NO2', 'O3', 'CH4', 'CO', 'ISOP', 'ACET', 'C2H6', 'C3H8', 'PRPE', 'ALK4',
             'MP', 'H2O2', 'QV', 'T', 'CLOUD',  'CH2O']
# Variables that will be written to a netcdf file for input into the GBRT model.
VarList = ['Lat', 'PL', 'T', 'NO2', 'O3', 'CH4', 'CO', 'ISOP', 'ACET', 'C2H6', 'C3H8', 'PRPE',
           'ALK4', 'MP', 'H2O2', 'TAUCLWDWN', 'TAUCLIDWN', 'TAUCLIUP', 'TAUCLWUP', 'CLOUD', 'QV',
           'GMISTRATO3', 'ALBUV', 'AODUP', 'AODDWN', 'CH2O', 'SZA', 'OH', 'trop_mask', 'aircol']
output = {}

for single_date in _daterange(start_date, end_date):
    print("Extracting variables for: " + str(single_date))
    merra2_dir = '/css/merra2gmi/pub/Y' + \
        str(single_date.year) + '/M' + f"{single_date.month:02}" + '/'
    # building the tropospheric mask
    TROPPB = _read_nc(merra2_dir + 'MERRA2_GMI.tavg24_2d_dad_Nx.' + str(single_date.year) +
                      f"{single_date.month:02}" + f"{single_date.day:02}" + '.nc4', 'TROPPB')
    PL = _read_nc(merra2_dir + 'MERRA2_GMI.tavg3_3d_met_Nv.' + str(single_date.year) +
                  f"{single_date.month:02}" + f"{single_date.day:02}" + '.nc4', 'PL')
    # converting to daily-averaged values
    PL = np.nanmean(PL, axis=0).squeeze()
    output["PL"] = PL/100.0
    trop_mask = np.zeros_like(PL)
    for z in range(0, np.shape(PL)[0]):
        diff_p = PL[z, :, :] - TROPPB[:, :]
        trop_mask[z, :, :] = diff_p >= 0.0
    trop_mask = np.multiply(trop_mask, 1.0)
    output["trop_mask"] = trop_mask
    # storing lat and lon for later
    Lat = np.arange(-90, 90.5, .5)
    Lon = np.arange(-180, 180, .625)
    # extracting OH values
    OH = _read_nc(merra2_dir + 'MERRA2_GMI.tavg24_3d_dac_Nv.' + str(single_date.year) +
                  f"{single_date.month:02}" + f"{single_date.day:02}" + '.nc4', 'OH')
    output["OH"] = OH
    # extracting values from InputList
    for var in InputList:
        # pinpointing the files
        print('......... Reading ' + str(var))
        if var in DACList:
            fname = merra2_dir + 'MERRA2_GMI.tavg24_3d_dac_Nv.' + str(single_date.year) +\
                f"{single_date.month:02}" + f"{single_date.day:02}" + '.nc4'
        if var in MetList:
            fname = merra2_dir + 'MERRA2_GMI.tavg3_3d_met_Nv.' + str(single_date.year) +\
                f"{single_date.month:02}" + f"{single_date.day:02}" + '.nc4'
        if var in CloudList:
            fname = merra2_dir + 'MERRA2_GMI.tavg3_3d_cld_Nv.' + str(single_date.year) +\
                f"{single_date.month:02}" + f"{single_date.day:02}" + '.nc4'
        if var in NxList:
            fname = merra2_dir + 'MERRA2_GMI.tavg24_2d_dad_Nx.' + str(single_date.year) +\
                f"{single_date.month:02}" + f"{single_date.day:02}" + '.nc4'
        if var == 'ALBUV':
            fname = './OMILER_345nm_M2GMIGrid.nc'
        if var in AODList:
            fname = merra2_dir + 'MERRA2_GMI.tavg24_3d_adf_Nv.' + str(single_date.year) +\
                f"{single_date.month:02}" + f"{single_date.day:02}" + '.nc4'
            fname_height_mid = merra2_dir + 'MERRA2_GMI.tavg3_3d_met_Nv.' + str(single_date.year) +\
                f"{single_date.month:02}" + f"{single_date.day:02}" + '.nc4'
            fname_height_edge = merra2_dir + 'MERRA2_GMI.tavg3_3d_mst_Ne.' + str(single_date.year) +\
                f"{single_date.month:02}" + f"{single_date.day:02}" + '.nc4'

        # extracting vars one by one
        if var == 'GMISTRATO3':
            GMISTRATO3 = _read_nc(fname, 'GMITO3') - _read_nc(fname, 'GMITTO3')
            output["GMISTRATO3"] = np.tile(GMISTRATO3, (72, 1, 1))
        if ((var == 'TAUCLIUP') or (var == 'TAUCLIDWN')):
            OpticalThickness = _read_nc(fname, 'TAUCLI')
            OpticalThickness = np.mean(OpticalThickness, axis=0).squeeze()
        if ((var == 'TAUCLWUP') or (var == 'TAUCLWDWN')):
            OpticalThickness = _read_nc(fname, 'TAUCLW')
            OpticalThickness = np.mean(OpticalThickness, axis=0).squeeze()
        if var in AODList:
            AeosolEF = _read_nc(fname, 'BCSCACOEF') +\
                _read_nc(fname, 'DUSCACOEF') +\
                _read_nc(fname, 'NISCACOEF') +\
                _read_nc(fname, 'OCSCACOEF') +\
                _read_nc(fname, 'SSSCACOEF') +\
                _read_nc(fname, 'SUSCACOEF')

            height_mid = _read_nc(fname_height_mid, 'H')
            height_edge = _read_nc(fname_height_edge, 'ZLE')
            # daily-averaging
            height_mid = np.nanmean(height_mid, axis=0).squeeze()
            height_edge = np.nanmean(height_edge, axis=0).squeeze()
            # thickness
            dh = -2.0*(height_edge[1:, :, :] - height_mid)
            # AOD
            OpticalThickness = dh*AeosolEF

        if ((var == 'TAUCLIUP') or (var == 'TAUCLWUP') or (var == 'AODUP')):
            OpticalThicknessUP = np.zeros_like(OpticalThickness)
            for z in range(0, np.shape(OpticalThickness)[0]):
                OpticalThicknessUP[z, :, :] = np.sum(
                    OpticalThickness[0:z, :, :], axis=0)
            output[var] = OpticalThicknessUP
        if ((var == 'TAUCLIDWN') or var == ('TAUCLWDWN') or (var == 'AODDWN')):
            OpticalThicknessDOWN = np.zeros_like(OpticalThickness)
            for z in range(0, np.shape(OpticalThickness)[0]):
                OpticalThicknessDOWN[z, :, :] = np.sum(
                    OpticalThickness[z:np.shape(OpticalThickness)[0]+1, :, :], axis=0)
            output[var] = OpticalThicknessDOWN
        if var == 'Lat':
            output[var] = np.tile(
                np.tile(Lat, (np.size(Lon), 1)).T, (72, 1, 1))
        if var == 'SZA':
            SZA = np.zeros((np.size(Lat), np.size(Lon)))
            for a in range(np.size(Lon)):
                for b in range(np.size(Lat)):
                    seconds_lon = (Lon[a]*3600)/15
                    times = pd.Timestamp(str(single_date.year) + '-' + f"{single_date.month:02}" +
                                         '-' + f"{single_date.day:02}" +
                                         ' 12:00:00')-pd.Timedelta(seconds=seconds_lon)
                    solar_param = _cal_SZA(times, Lat[b], Lon[a], 0.0)
                    SZA[b, a] = np.array(solar_param.zenith)
            # repeating SZA in 72 layers
            output["SZA"] = np.tile(SZA, (72, 1, 1))
        if var == 'ALBUV':
            LER = _read_nc(fname, 'LER')
            LER = LER[single_date.month-1, :, :].squeeze()
            sizePL = np.shape(LER)
            output[var] = np.tile(LER, (72, 1, 1))
        if var == 'aircol':
            DELP = _read_nc(fname, 'DELP')
            DELP = np.mean(DELP, axis=0).squeeze()
            output[var] = DELP/g/Mair*N_A
        # this else deals with a lot of variables including temperature, CO, NO2, ...
        if var in InputAsIs:
            species = _read_nc(fname, var)
            if np.ndim(species) == 4:  # variables such as Qv, T need to be converted to daily
                species = np.nanmean(species, axis=0).squeeze()
            output[var] = species

    # saving all vars in outputs in a netcdf file
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = 'MERRA2_GMI_XGBOOST_Inputs_' + str(single_date.year) +\
        f"{single_date.month:02}" + f"{single_date.day:02}"

    ncfile = Dataset(output_folder + '/' + output_file + '.nc', 'w')
    # create the x and y dimensions.
    ncfile.createDimension('z', np.shape(height_mid)[0])
    ncfile.createDimension('x', np.shape(height_mid)[1])
    ncfile.createDimension('y', np.shape(height_mid)[2])

    for var in VarList:
        print('......... Writing ' + str(var))
        data = output[var]
        if np.ndim(data) == 2:
            data_nc = ncfile.createVariable(
                var, np.dtype('float32').char, ('x', 'y'))
            data_nc[:, :] = data
        if np.ndim(data) == 3:
            data_nc = ncfile.createVariable(
                var, np.dtype('float32').char, ('z', 'x', 'y'))
            data_nc[:, :, :] = data
    ncfile.close()
