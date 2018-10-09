import os
import argparse
from modules import io
from modules import vascular_data as sv
import scipy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('global_config_file')
parser.add_argument('case_config_file')

args = parser.parse_args()

global_config_file = os.path.abspath(args.global_config_file)
case_config_file = os.path.abspath(args.case_config_file)


global_config = io.load_yaml(global_config_file)
case_config   = io.load_yaml(case_config_file)

####################################
# Get necessary params
####################################
cases = os.listdir(global_config['CASES_DIR'])
cases = [global_config['CASES_DIR']+'/'+f for f in cases if 'case.' in f]

spacing_vec = [case_config['SPACING']]*2
dims_vec    = [case_config['DIMS']]*2
ext_vec     = [case_config['DIMS']-1]*2
path_start  = case_config['PATH_START']

files = open(case_config['DATA_DIR']+'/files.txt','w')

for i, case_fn in enumerate(cases):
    case_dict = io.load_yaml(case_fn)
    print(case_dict['NAME'])

    image_dir = case_config['DATA_DIR']+'/'+case_dict['NAME']
    sv.mkdir(image_dir)

    image        = sv.read_mha(case_dict['IMAGE'])
    #image        = sv.resample_image(image,case_config['SPACING'])

    segmentation = sv.read_mha(case_dict['SEGMENTATION'])
    #segmentation = sv.resample_image(segmentation,case_config['SPACING'])

    path_dict    = sv.parsePathFile(case_dict['PATHS'])
    group_dir    = case_dict['GROUPS']

    im_np  = sv.vtk_image_to_numpy(image)
    seg_np = sv.vtk_image_to_numpy(segmentation)
    blood_np = im_np[seg_np>0.1]

    stats = {"MEAN":np.mean(im_np), "STD":np.std(im_np), "MAX":np.amax(im_np),
    "MIN":np.amin(im_np),
    "BLOOD_MEAN":np.mean(blood_np),
    "BLOOD_STD":np.std(blood_np),
    "BLOOD_MAX":np.amax(blood_np),
    "BLOOD_MIN":np.amin(blood_np)}

    for grp_id in path_dict.keys():
        path_info      = path_dict[grp_id]
        path_points    = path_info['points']
        group_name     = path_info['name']
        group_filename = group_dir +'/'+group_name

        if not os.path.exists(group_filename): continue

        group_dict = sv.parseGroupFile(group_filename)

        group_points = sorted(group_dict.keys())

        if len(group_points) < 4: continue

        tup = sv.get_segs(path_points,group_dict,
            [case_config['DIMS']]*2, [case_config['SPACING']]*2,
            case_config['NUM_CONTOUR_POINTS'])

        if tup == None: continue

        group_data_dir = image_dir+'/'+group_name
        sv.mkdir(group_data_dir)

        segs,norm_grps,interp_grps,means = tup

        im_slices  = []
        seg_slices = []

        if not 'SEG_TYPE' in case_config:
            for i,I in enumerate(group_points[path_start:-path_start]):
                j = i+path_start

                v = path_points[I]
                im_slice = sv.getImageReslice(image, ext_vec,
                    v[:3],v[3:6],v[6:9], case_config['SPACING'],True)
                seg_slice = sv.getImageReslice(segmentation, ext_vec,
                    v[:3],v[3:6],v[6:9], case_config['SPACING'],True)

                try:
                    x_path  = '{}/{}.X.npy'.format(group_data_dir,I)
                    y_path  = '{}/{}.Y.npy'.format(group_data_dir,I)
                    yc_path = '{}/{}.Yc.npy'.format(group_data_dir,I)
                    c_path  = '{}/{}.C.npy'.format(group_data_dir,I)
                    ci_path =  '{}/{}.C_interp.npy'.format(group_data_dir,I)

                    yaml_path = '{}/{}.yaml'.format(group_data_dir,I)

                    radius = np.sqrt((1.0*np.sum(segs[j]))/np.pi)

                    yaml_dict = {}
                    yaml_dict['X'] = x_path
                    yaml_dict['Y'] = y_path
                    yaml_dict['Yc'] = yc_path
                    yaml_dict['C'] = c_path
                    yaml_dict['C_interp'] = ci_path
                    yaml_dict['point'] = I
                    yaml_dict['path_name'] = group_name
                    yaml_dict['path_id'] = grp_id
                    yaml_dict['image'] = case_dict['NAME']
                    yaml_dict['extent'] = case_config['DIMS']
                    yaml_dict['dimensions'] = case_config['CROP_DIMS']
                    yaml_dict['spacing'] = case_config['SPACING']
                    yaml_dict['radius'] = float(radius)
                    for k,v in stats.items():
                        yaml_dict[k] = float(v)

                    io.save_yaml(yaml_path, yaml_dict)

                    np.save(x_path, im_slice)
                    np.save(y_path, seg_slice)
                    np.save(yc_path, segs[j])
                    np.save(c_path, norm_grps[j])
                    np.save(ci_path, interp_grps[j])

                    scipy.misc.imsave('{}/{}.X.png'.format(group_data_dir,I),im_slice)
                    scipy.misc.imsave('{}/{}.Y.png'.format(group_data_dir,I),seg_slice)
                    scipy.misc.imsave('{}/{}.Yc.png'.format(group_data_dir,I),segs[j])

                    files.write(yaml_path+'\n')
                except:
                    print( "failed to save {}/{}".format(group_data_dir,I))
        elif case_config['SEG_TYPE'] == 'LOFT':

            image_dir = case_config['DATA_DIR']+'/'+case_dict['NAME']+'_loft'
            sv.mkdir(image_dir)
            group_data_dir = image_dir+'/'+group_name
            sv.mkdir(group_data_dir)

            lofted_segs, lofted_groups = sv.loft_path_segs(interp_grps,means,
                group_dict, dims_vec,spacing_vec)

            for i in range(group_points[path_start],group_points[-path_start]):
                j = i

                v = path_points[i]
                im_slice = sv.getImageReslice(image, ext_vec,
                    v[:3],v[3:6],v[6:9],True)
                seg_slice = sv.getImageReslice(segmentation, ext_vec,
                    v[:3],v[3:6],v[6:9],True)

                try:
                    np.save('{}/{}.X.npy'.format(group_data_dir,i),im_slice)
                    np.save('{}/{}.Y.npy'.format(group_data_dir,i),seg_slice)
                    np.save('{}/{}.Yc.npy'.format(group_data_dir,i),lofted_segs[i])
                    np.save('{}/{}.C.npy'.format(group_data_dir,i),lofted_groups[i])

                    scipy.misc.imsave('{}/{}.X.png'.format(group_data_dir,i),im_slice)
                    scipy.misc.imsave('{}/{}.Y.png'.format(group_data_dir,i),seg_slice)
                    scipy.misc.imsave('{}/{}.Yc.png'.format(group_data_dir,i),lofted_segs[i])

                    files.write('{}/{}\n'.format(group_data_dir,i))
                except:
                    print( "failed to save {}/{}".format(group_data_dir,i))


        elif case_config['SEG_TYPE'] == '3D':
            num_steps = case_config['NUM_STEPS']
            pixel_step = case_config['PIXEL_STEP']
            step_range = np.arange(-num_steps, num_steps)

            for i,I in enumerate(group_points[path_start:-path_start]):
                j = i+path_start

                v = path_points[I]
                p   = v[:3]
                tan = v[3:6]
                tx  = v[6:9]

                im_slices  = []
                seg_slices = []

                for step in step_range:
                    p_ = p + step*pixel_step*case_config['SPACING']*tan

                    im = sv.getImageReslice(image, ext_vec,
                        p_,tan,tx, case_config['SPACING'],True)
                    seg = sv.getImageReslice(segmentation, ext_vec,
                        p_,tan,tx, case_config['SPACING'],True)

                    im_slices.append(im)
                    seg_slices.append(seg)
                try:
                    x_path  = '{}/{}.X.npy'.format(group_data_dir,I)
                    y_path  = '{}/{}.Y.npy'.format(group_data_dir,I)
                    yc_path = '{}/{}.Yc.npy'.format(group_data_dir,I)
                    c_path  = '{}/{}.C.npy'.format(group_data_dir,I)
                    ci_path =  '{}/{}.C_interp.npy'.format(group_data_dir,I)

                    yaml_path = '{}/{}.yaml'.format(group_data_dir,I)

                    radius = np.sqrt((1.0*np.sum(segs[j]))/np.pi)

                    yaml_dict = {}
                    yaml_dict['X'] = x_path
                    yaml_dict['Y'] = y_path
                    yaml_dict['Yc'] = yc_path
                    yaml_dict['C'] = c_path
                    yaml_dict['C_interp'] = ci_path
                    yaml_dict['point'] = I
                    yaml_dict['path_name'] = group_name
                    yaml_dict['path_id'] = grp_id
                    yaml_dict['image'] = case_dict['NAME']
                    yaml_dict['extent'] = case_config['DIMS']
                    yaml_dict['dimensions'] = case_config['CROP_DIMS']
                    yaml_dict['spacing'] = case_config['SPACING']
                    yaml_dict['radius'] = float(radius)
                    for k,v in stats.items():
                        yaml_dict[k] = float(v)

                    io.save_yaml(yaml_path, yaml_dict)

                    np.save(x_path, np.array(im_slices))
                    np.save(y_path, np.array(seg_slices))
                    np.save(yc_path, segs[j])
                    np.save(c_path, norm_grps[j])
                    np.save(ci_path, interp_grps[j])


                    for k in range(len(im_slices)):
                        scipy.misc.imsave('{}/{}.X.{}.png'.format(group_data_dir,I,k),im_slices[k])
                        scipy.misc.imsave('{}/{}.Y.{}.png'.format(group_data_dir,I,k),seg_slices[k])
                    scipy.misc.imsave('{}/{}.Yc.png'.format(group_data_dir,I),segs[j])

                    files.write(yaml_path+'\n')
                except:
                    print( "failed to save {}/{}".format(group_data_dir,I))

        io.write_csv(image_dir+'/'+'image_stats.csv',stats)

files.close()
