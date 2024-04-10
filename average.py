import numpy as np
from netCDF4 import Dataset
import warnings
import glob
import yaml
from scipy.io import savemat
import datetime

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
# Read the control file
with open('./control.yml', 'r') as stream:
    try:
        ctrl_opts = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Exception(exc)

output_dir = ctrl_opts['output_dir']
var_perturb = ctrl_opts['var_perturb']
startdate = ctrl_opts['start_date']
enddate = ctrl_opts['end_date']

# convert dates to datetime
start_date = datetime.date(int(startdate[0:4]), int(
    startdate[5:7]),int(startdate[8:10]))
end_date = datetime.date(int(enddate[0:4]), int(
    enddate[5:7]),int(startdate[8:10]))

list_months = []
list_years = []
for single_date in _daterange(start_date, end_date):
    list_months.append(single_date.month)
    list_years.append(single_date.year)

output = {}
var_perturb.append('org')

for var in var_perturb:
    OH_pred = np.zeros((361,576,
        len(range(np.min(list_months),
                  np.max(list_months)+1)),
        len(range(np.min(list_years), np.max(list_years)+1))))
    sens = np.zeros_like(OH_pred)
    sens_full = np.zeros((361,576,
        len(range(np.min(list_months),
                  np.max(list_months)+1)),
        len(range(np.min(list_years), np.max(list_years)+1))))
    for year in range(np.min(list_years), np.max(list_years)+1):
        for month in range(np.min(list_months), np.max(list_months)+1):
            files = sorted(glob.glob(output_dir + '/*' + '_' + str(var) +'_' + '*' + str(year) + f"{month:02}" + '*'))
            OH_pred_chosen = []
            sens_up_chosen = []
            sens_down_chosen = []
            sens_full_up_chosen = []
            sens_full_down_chosen = []
            for f in files:
                if var == 'org':
                   print(f)
                   OH_pred_chosen.append(_read_nc(f,'OH_pred'))
                else:
                   if "_up_" in f:
                       print(f)
                       sens_up_chosen.append(_read_nc(f,'OH_pred'))
                       sens_full_up_chosen.append(_read_nc(f,'OH_pred_full'))
                   elif "_down_" in f:
                       print(f)
                       sens_down_chosen.append(_read_nc(f,'OH_pred'))
                       sens_full_down_chosen.append(_read_nc(f,'OH_pred_full'))
            if var == 'org':
               OH_pred_chosen = np.mean(np.array(OH_pred_chosen),axis=0)
               OH_pred[:,:,month - min(list_months), year - min(
                   list_years)] = OH_pred_chosen
            else:
               sens_up_chosen = np.mean(np.array(sens_up_chosen),axis=0)
               sens_down_chosen = np.mean(np.array(sens_down_chosen),axis=0)
               sens_full_up_chosen = np.mean(np.array(sens_full_up_chosen),axis=0)
               sens_full_down_chosen = np.mean(np.array(sens_full_down_chosen),axis=0)
               sens[:,:,month - min(list_months), year - min(
                   list_years)] = (sens_up_chosen-sens_down_chosen)/0.20
               sens_full[:,:,month - min(list_months), year - min(
                   list_years)] = (sens_full_up_chosen[-1,:,:].squeeze()-sens_full_down_chosen[-1,:,:].squeeze())/0.20
    if var == 'org':
       output[str(var) + '_OH'] = OH_pred
    else:
       output[str(var) + '_OH'] = sens
       output[str(var) + '_full_OH'] = sens_full

savemat("OH_sens_2012.mat", output)
