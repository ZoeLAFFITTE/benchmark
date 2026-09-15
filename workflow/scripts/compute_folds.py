import sys
import numpy as np
from mindboggle.features.folds import find_depth_threshold, extract_folds

input_vtk = sys.argv[1]
output_vtk = sys.argv[2]
output_npy = sys.argv[3]

depth_threshold, bins, bin_edges = find_depth_threshold(
    depth_file=input_vtk,
    min_vertices=10000,
    verbose=True)

folds, n_folds, folds_file = extract_folds(
    depth_file=input_vtk,
    depth_threshold=depth_threshold,
    min_fold_size=50,
    save_file=True,
    output_file=output_vtk,
    background_value=-1,
    verbose=True)

folds = np.array(folds, dtype=int)
np.save(output_npy, folds)