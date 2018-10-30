import subprocess
import os
from modules import io

dir_  = './data/cases'
cases = os.listdir(dir_)
config_file = './config/global.yaml'

config = io.load_yaml(config_file)

f = open(config['DATA_DIR']+'/files.txt','w')
f.close()

for c in cases:
    f = dir_+'/'+c
    print(f)
    subprocess.check_call('python process_data.py {} {}'.format(config_file, f), shell=True)
