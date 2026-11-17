import sys
import numpy as np
from mindboggle.features.folds import find_depth_threshold, extract_folds
from mindboggle.features.fundi import extract_fundi

curv_pial_vtk = sys.argv[1]
depth_pial_vtk = sys.argv[2]
depth_pial_vtk_res = sys.argv[3]
output_folds_vtk = sys.argv[4]
output_fundi_vtk = sys.argv[5]

depth_threshold, bins, bin_edges = find_depth_threshold(
    depth_file=depth_pial_vtk,
    min_vertices=10000,
    verbose=True)

folds, n_folds, folds_file = extract_folds(
    depth_file=depth_pial_vtk,
    depth_threshold=depth_threshold,
    min_fold_size=50,
    save_file=True,
    output_file=output_folds_vtk,
    background_value=-1,
    verbose=True)

fundus_per_fold, n_fundi, fundi_file = extract_fundi(
    folds=folds,
    curv_file=curv_pial_vtk,
    depth_file=depth_pial_vtk_res,
    save_file=True,
    output_file=output_fundi_vtk,
    background_value=-1,
    verbose=True)

# folds = np.array(folds, dtype=int)
# np.save(output_npy, folds)