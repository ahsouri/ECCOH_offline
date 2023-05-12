import xgboost as xgb
import numpy as np
from netCDF4 import Dataset
import warnings
import glob
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)


input_folder = '/home/asouri/git_repos/mule/offline_eccoh/'
input_files = sorted(glob.glob(input_folder + '/*.nc'))
output_folder = ''
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

VarList = ['Lat', 'PL', 'T', 'NO2', 'O3', 'CH4', 'CO', 'ISOP', 'ACET', 'C2H6', 'C3H8', 'PRPE',
           'ALK4', 'MP', 'H2O2', 'TAUCLWDWN', 'TAUCLIDWN', 'TAUCLIUP', 'TAUCLWUP', 'CLOUD', 'QV',
           'GMISTRATO3', 'ALBUV', 'AODUP', 'AODDWN', 'CH2O', 'SZA', 'OH', 'trop_mask']

input = {}

for fname in input_files:
    for var in VarList:
        input[var] = _read_nc(fname, var)

    # applying the tropospheric mask
    mask_trop = input["trop_mask"]
    indices_legit = np.argwhere(mask_trop == 1.0)
    xgb_input = np.zeros((np.size(indices_legit), len(VarList)-2))
    counter = 0
    for var in VarList:
        temp_var = mask_trop*input[var]
        temp_var[temp_var == 0.0] = np.nan
        xgb_input[:, counter] = temp_var[~np.isnan(temp_var)]
        counter += 1

    # predict OH
    month = (fname.split('_')[-1])[4:6]
    FileModifier = 'UpDwnALBUVSZAAll_NoGMIALB_NoScale_NoRegressor_NewXGB_M' + month
    modelname = '/discover/nobackup/dcanders/QuickChem/Data/RTModels/' + \
        'xgboh_' + FileModifier + '.model'
    bst = xgb.Booster({'nthread': 4})
    bst.load_model(modelname)
    Ypred = bst.predict(xgb.DMatrix(xgb_input))
    Ypred = 10**Ypred

    # reshape Ypred to become similar to OH
    OH_org = input["OH"]
    OH_pred = np.zeros_like(OH_org)
    OH_pred[indices_legit] = Ypred

    # converting OH to tropospheric OH
    T = input["T"]
    CH4 = input["CH4"]
    aircol = _read_nc(fname, 'aircol')
    MCH4 = CH4*aircol/6.02214076e23/16.04e3  # kg of MCH4
    K_OH_CH4 = 1.85e-12*np.exp(-1690/T)
    numerator_pred = np.sum(OH_pred*MCH4*K_OH_CH4, axis=0).squeeze()
    denominator_pred = np.sum(MCH4*K_OH_CH4, axis=0).squeeze()
    numerator_org = np.sum(OH_org*MCH4*K_OH_CH4, axis=0).squeeze()
    denominator_pred = np.sum(MCH4*K_OH_CH4, axis=0).squeeze()
    OHtrop_org = numerator_org/denominator_pred
    OHtrop_pred = numerator_pred/denominator_pred

    output_file = 'MERRA2_GMI_XGBOOST_Output_' + fname.split('_')[-1]

    ncfile = Dataset(output_folder + '/' + output_file, 'w')
    # create the x and y dimensions.
    ncfile.createDimension('x', np.shape(input["PL"])[1])
    ncfile.createDimension('y', np.shape(input["PL"])[2])

    data_nc = ncfile.createVariable(
        'OH_org', np.dtype('float32').char, ('x', 'y'))
    data_nc[:, :] = OHtrop_org

    data_nc = ncfile.createVariable(
        'OH_pred', np.dtype('float32').char, ('x', 'y'))
    data_nc[:, :] = OHtrop_pred

    ncfile.close()
