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


def _cal_trop_OH(input, OH):
    # converting OH to tropospheric OH
    T = input["T"]
    PL = input["PL"]
    CH4 = input["CH4"]
    aircol = input["aircol"]
    N_A = 6.02214076e23
    MCH4 = CH4*aircol/N_A*16.04e-3  # kg of MCH4
    K_OH_CH4 = 1.85e-12*np.exp(-1690/T)
    R = 8.314e4  # cm^3 mbar /K /mol
    M = N_A*PL/R/T
    numerator = np.sum(M*OH*MCH4*K_OH_CH4, axis=0).squeeze()
    denominator = np.sum(MCH4*K_OH_CH4, axis=0).squeeze()
    return 1e6*numerator/denominator


def run_eccoh(date:str,var_perturb: list, pertubation: float, output_folder: str):
    input_files = ['./inputs/MERRA2_GMI_XGBOOST_Inputs_' + date[0:4] + date[5:7] + date[8:10] + '.nc']

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    VarList = ['Lat', 'PL', 'T', 'NO2', 'O3', 'CH4', 'CO', 'ISOP', 'ACET', 'C2H6', 'C3H8', 'PRPE',
               'ALK4', 'MP', 'H2O2', 'TAUCLWDWN', 'TAUCLIDWN', 'TAUCLIUP', 'TAUCLWUP', 'CLOUD', 'QV',
               'GMISTRATO3', 'ALBUV', 'AODUP', 'AODDWN', 'CH2O', 'SZA', 'OH', 'trop_mask', 'aircol']

    input = {}

    for fname in input_files:
        print("Reading the input file from " + fname)
        for var in VarList:
            input[var] = _read_nc(fname, var)

        # perturb the target variable
        if var_perturb:
            input[var_perturb] = input[var_perturb]*pertubation
        print("Reading Completed")
        mask_trop = input["trop_mask"]
        indices_legit = np.argwhere(mask_trop == 1.0)
        xgb_input = np.zeros((np.shape(indices_legit)[0], len(VarList)-3))
        counter = 0
        for var in VarList:
            mask_trop[mask_trop != 1.0] = np.nan
            temp_var = input[var]*mask_trop
            xgb_input[:, counter] = temp_var[~np.isnan(temp_var)]
            counter += 1
            if counter == 27:
                break
        print("OH prediction begins")
        # predict OH
        month = (fname.split('_')[-1])[4:6]
        FileModifier = 'UpDwnALBUVSZAAll_NoGMIALB_NoScale_NoRegressor_NewXGB_M' + month
        modelname = '/discover/nobackup/dcanders/QuickChem/Data/RTModels/' + \
            'xgboh_' + FileModifier + '.model'
        bst = xgb.Booster({'nthread': 4})
        bst.load_model(modelname)
        Ypred = bst.predict(xgb.DMatrix(xgb_input))
        Ypred = 10**Ypred
        print("OH prediction is done, writing out the results")
        # reshape Ypred to OH_org
        OH_org = input["OH"]
        OH_pred = np.zeros_like(OH_org)
        for i in range(0, np.shape(indices_legit)[0]):
            OH_pred[indices_legit[i, 0], indices_legit[i, 1],
                    indices_legit[i, 2]] = Ypred[i]

        # converting OH to tropospheric OH
        OHtrop_org = _cal_trop_OH(input, OH_org)
        OHtrop_pred = _cal_trop_OH(input, OH_pred)

        # writing to a ncfile
        if var_perturb:
            if pertubation > 1.0:
                output_file = 'MERRA2_GMI_XGBOOST_output_' + \
                    str(var_perturb) + '_up_' + fname.split('_')[-1]
            else:
                output_file = 'MERRA2_GMI_XGBOOST_output_' + \
                    str(var_perturb) + '_down_' + fname.split('_')[-1]
        else:
            output_file = 'MERRA2_GMI_XGBOOST_org_' + fname.split('_')[-1]
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
