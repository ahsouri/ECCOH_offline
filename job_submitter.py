import yaml
import os
import datetime
import numpy as np


def _daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days+1)):
        yield start_date + datetime.timedelta(n)


# Read the control file
with open('./control.yml', 'r') as stream:
    try:
        ctrl_opts = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Exception(exc)

startdate = ctrl_opts['start_date']
enddate = ctrl_opts['end_date']
python_bin = ctrl_opts['python_bin']
debug_on = ctrl_opts['debug']

# convert dates to datetime
start_date = datetime.date(int(startdate[0:4]), int(
    startdate[5:7]),int(startdate[8:10]))
end_date = datetime.date(int(enddate[0:4]), int(
    enddate[5:7]),int(startdate[8:10]))
list_months = []
list_years = []


# submit jobs per month per year (12 jobs per year)
if not os.path.exists('./jobs'):
    os.makedirs('./jobs')

for single_date in _daterange(start_date, end_date):
    # slurm command
    # Opening a file
    file = open('./jobs/' + 'job_' + str(single_date) + '.j', 'w')
    slurm_cmd = '#!/bin/bash \n'
    slurm_cmd += '#SBATCH -J eccoh_offline \n'
    slurm_cmd += '#SBATCH --account=s1043 \n'
    slurm_cmd += '#SBATCH --ntasks=1 \n'
    slurm_cmd += '#SBATCH --cpus-per-task=1' + ' \n'
    slurm_cmd += '#SBATCH --mem=20G \n'
    if debug_on:
        slurm_cmd += '#SBATCH --qos=debug \n'
    else:
        slurm_cmd += '#SBATCH -t 0:30:00 \n'
    slurm_cmd += '#SBATCH -o eccoh-%j.out \n'
    slurm_cmd += '#SBATCH -e eccoh-%j.err \n'
    slurm_cmd += python_bin + ' ./job.py ' + str(single_date)
    file.writelines(slurm_cmd)
    file.close()
    os.system('sbatch ' + './jobs/' + 'job_' + str(single_date) + '.j')