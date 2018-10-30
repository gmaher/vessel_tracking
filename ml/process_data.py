import os
import argparse
from modules import io
from modules import vascular_data as sv
import scipy
import numpy as np
from tqdm import tqdm

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

spacing_vec = [global_config['SPACING']]*3
dims_vec    = [global_config['DIM']]*2
ext_vec     = [global_config['DIM']-1]*2
DIM = global_config['DIM']

files = open(global_config['DATA_DIR']+'/files.txt','a')


print(case_dict['NAME'])

image_dir = global_config['DATA_DIR']+'/'+case_dict['NAME']
sv.mkdir(image_dir)

image        = sv.sitk_read_image(case_dict['IMAGE'])

print(image.GetSize())
image        = sv.sitk_resample_image(image,spacing_vec)

segmentation = sv.sitk_read_image(case_dict['SEGMENTATION'])
segmentation = sv.sitk_resample_image(segmentation,spacing_vec)

im_np  = sv.sitk_image_to_numpy(image)
seg_np = sv.sitk_image_to_numpy(segmentation)
seg_np = (1.0*seg_np-np.amin(seg_np))/(np.amax(seg_np)-np.amin(seg_np)+1e-3)

H,W,D = seg_np.shape
WS = global_config['SMALL_WINDOW_DIM']
print(image.GetSize())
ids = []
small_ids = []

ids_neg = []

for i in tqdm(range(DIM,H-DIM,3)):
    for j in range(DIM,W-DIM,3):
        for k in range(DIM,D-DIM,3):
            if seg_np[i,j,k] > 0.1:

                vpix = np.sum( seg_np[i, int(j-WS/2):int(j+WS/2), int(k-WS/2):int(k+WS/2) ])
                if vpix < global_config['SMALL_VESSEL_CUTOFF']:
                    small_ids.append((i,j,k))
                else:
                    ids.append((i,j,k))

            d = int(DIM/2)

            y = seg_np[i, j-d:j+d, k-d:k+d]
            s = np.sum(y)
            if s<5:
                ids_neg.append((i,j,k))

#Sample positive images
pos_dir = image_dir+'/positive'
sv.mkdir(pos_dir)

np.random.shuffle(ids)
np.random.shuffle(small_ids)

print("small vessels: {}".format(len(small_ids)))
print("large_vessels: {}".format(len(ids)))

if len(small_ids) < global_config['DATA_SAMPLES']/2:
    n = global_config['DATA_SAMPLES'] - len(small_ids)
    ids = ids[:n] + small_ids
else:
    n = int(global_config['DATA_SAMPLES']/2)
    ids = ids[:n] + small_ids[:n]

for i,t in tqdm(enumerate(ids)):
    if (i > global_config['DATA_SAMPLES']): break

    d = int(DIM/2)
    x = im_np[t[0], t[1]-d:t[1]+d, t[2]-d:t[2]+d]
    y = seg_np[t[0], t[1]-d:t[1]+d, t[2]-d:t[2]+d]

    files.write(pos_dir+'/{}\n'.format(i))

    np.save(pos_dir+'/{}.x.npy'.format(i),x)
    np.save(pos_dir+'/{}.y.npy'.format(i),y)

    try:
        scipy.misc.imsave(pos_dir+'/{}.x.png'.format(i),x)
        scipy.misc.imsave(pos_dir+'/{}.y.png'.format(i),y)

    except:
        pass

#Sample negative images
neg_dir = image_dir+'/negative'
sv.mkdir(neg_dir)

np.random.shuffle(ids_neg)
for i,t in tqdm(enumerate(ids_neg)):
    if (i > global_config['DATA_SAMPLES']): break

    d = int(DIM/2)
    x = im_np[t[0], t[1]-d:t[1]+d, t[2]-d:t[2]+d]
    y = seg_np[t[0], t[1]-d:t[1]+d, t[2]-d:t[2]+d]

    files.write(neg_dir+'/{}\n'.format(i))

    np.save(neg_dir+'/{}.x.npy'.format(i),x)
    np.save(neg_dir+'/{}.y.npy'.format(i),y)

    try:
        scipy.misc.imsave(neg_dir+'/{}.x.png'.format(i),x)
        scipy.misc.imsave(neg_dir+'/{}.y.png'.format(i),y)
    except:
        pass

files.close()
