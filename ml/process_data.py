import os
import argparse
from modules import io
from modules import vascular_data as sv
import scipy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('global_config_file')
parser.add_argument('case_file')

args = parser.parse_args()

global_config_file = os.path.abspath(args.global_config_file)

global_config = io.load_yaml(global_config_file)

case_file = os.path.abspath(args.case_file)
case_dict = io.load_yaml(case_file)

####################################
# Get necessary params
####################################

spacing_vec = [global_config['SPACING']]*2
dims_vec    = [global_config['DIMS']]*2
ext_vec     = [global_config['DIMS']-1]*2

files = open(global_config['DATA_DIR']+'/files.txt','w')

print(case_dict['NAME'])

image_dir = global_config['DATA_DIR']+'/'+case_dict['NAME']
sv.mkdir(image_dir)

image        = sv.read_mha(case_dict['IMAGE'])
image        = sv.resample_image(image,global_config['SPACING'])

segmentation = sv.read_mha(case_dict['SEGMENTATION'])
segmentation = sv.resample_image(segmentation,global_config['SPACING'])

im_np  = sv.vtk_image_to_numpy(image)
seg_np = sv.vtk_image_to_numpy(segmentation)
