import sys
import numpy as np
from mindboggle.features.fundi import extract_fundi

input_folds_npy = sys.argv[1]
input_curv = sys.argv[2]
input_depth_rescaled = sys.argv[3]
output_fundi_vtk = sys.argv[4]


folds = np.load(input_folds_npy)
fundus_per_fold, n_fundi, fundi_file = extract_fundi(
    folds=folds,
    curv_file=input_curv,
    depth_file=input_depth_rescaled,
    save_file=True,
    output_file=output_fundi_vtk,
    background_value=-1,
    verbose=True)