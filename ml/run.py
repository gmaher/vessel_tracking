import subprocess
import os

dir_  = './data/cases'
cases = os.listdir(dir_)
config_file = './config/global.yaml'

for c in cases:
    f = dir_+'/'+c
    print(f)
    subprocess.check_call('python process_data.py {} {}'.format(config_file, f), shell=True)
