import numpy as np
import pandas as pd
import slam.io as sio
import slam.texture as stex

def match_basins(test_basins, retest_basins):
    """
    Compute the overlap between the basins of two surfaces after interpolation onto a reference mesh.

    :param test_basins: basins of the 1st surface (nd array)
    :param retest_basins: basins of the 2nd surface (nd array)
    :return: correspondence table. For each basin of the 1st surface, the best match on the 2nd surface and the overlap measurement.
    """
    t_labels = np.unique(test_basins)
    rt_labels = np.unique(retest_basins)

    correspondence = {}

    for t_lab in t_labels:
        # get the indices of vertices corresponding to the label t_lab
        inds = set(np.where(test_basins == t_lab)[0])
        max_overlap = -1
        second_max_overlap = -1
        for rt_lab in rt_labels:
            # get the indices of vertices corresponding to the label rt_lab
            corr_inds = set(np.where(retest_basins == rt_lab)[0])
            # compute the overlap as intersection/union of vertices
            lab_overlap = len(inds.intersection(corr_inds)) / len(inds.union(corr_inds))
            # keep only the two highest results
            if lab_overlap > max_overlap:
                max_overlap = lab_overlap
                best_match = rt_lab
            if max_overlap > lab_overlap > second_max_overlap:
                second_max_overlap = lab_overlap
        correspondence[int(t_lab)] = [int(best_match), max_overlap, second_max_overlap]

    return correspondence

# ------------------------------------------------------------------
# Inputs / outputs
# ------------------------------------------------------------------
basins_test_path=snakemake.input.basins_test_gii
basins_retest_path=snakemake.input.basins_retest_gii

basins_test_resampled_path=snakemake.input.basins_test_resampled_gii
basins_retest_resampled_path=snakemake.input.basins_retest_resampled_gii


sphere_reg_test_path=snakemake.input.sphere_reg_test_gii
sphere_reg_retest_path=snakemake.input.sphere_reg_retest_gii
sphere_template_path=snakemake.input.sphere_template_gii

# Load meshes
sphere_reg_test = sio.load_mesh(sphere_reg_test_path)
sphere_reg_retest = sio.load_mesh(sphere_reg_retest_path)
sphere_template = sio.load_mesh(sphere_template_path)

# Load texture
basins_test = sio.load_texture(basins_test_path).darray[0]
basins_retest = sio.load_texture(basins_retest_path).darray[0]

basins_test_resampled = sio.load_texture(basins_test_resampled_path).darray[0]
basins_retest_resampled = sio.load_texture(basins_retest_resampled_path).darray[0]

print(len(basins_test))
print(len(basins_retest))

print(len(basins_test_resampled))
print(len(basins_retest_resampled))

print(len(sphere_reg_test.vertices))
print(len(sphere_reg_retest.vertices))
print(len(sphere_template.vertices))

# 120809
# 120594
# 120809
# 120594
# 32492

print(np.array_equal(basins_test_resampled, basins_retest_resampled))
print(np.sum(basins_test_resampled != basins_retest_resampled))
print(np.unique(basins_test))
print(np.unique(basins_retest))

matches = match_basins(basins_test_resampled, basins_retest_resampled)
print(matches)

df_matches = pd.DataFrame.from_dict(
    matches,
    orient="index",
    columns=["retest_basin", "overlap", "second_overlap"],
)

df_matches.index.name = "test_basin"

df_matches.to_csv(
    snakemake.output.dice_tsv,
    sep="\t",
    index=True,
)