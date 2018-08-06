import json
import numpy as np
import SimpleITK as sitk

def load_json(fn):
    with open(fn,'r') as f:
        return json.load(f)

def write_json(d,fn):
    with open(fn,'w') as f:
        json.dump(d,f, indent=2)

def load_image(fn):
    if '.npy' in fn:
        return np.load(fn)
    if '.mha' in fn:
        return load_mha(fn)

def load_mha(fn):
    im = sitk.ReadImage(fn)
    return sitk.GetArrayFromImage(im)
