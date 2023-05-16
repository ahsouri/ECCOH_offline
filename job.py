import yaml
import sys
from get_inputs import get_eccoh_inputs
from run_xgboost import run_eccoh
import os.path

# Read the control file
with open('./control.yml', 'r') as stream:
    try:
        ctrl_opts = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Exception(exc)

output_dir = ctrl_opts['output_dir']
var_perturb = ctrl_opts['var_perturb']
ctm_dir = ctrl_opts['ctm_dir']
date = str(sys.argv[1])
# creating inputs
file = './inputs/MERRA2_GMI_XGBOOST_Inputs_' + date[0:4] + date[5:7] + date[8:10] + '.nc'
if os.path.isfile(file):
   pass
else:
   get_eccoh_inputs(date,date,ctm_dir)
# running eccoh without perturbation
run_eccoh(date,[],[],output_dir)
# running eccoh with +-10 perturbation
for var in var_perturb:
    run_eccoh(date,var,1.1,output_dir)
    run_eccoh(date,var,0.9,output_dir)
