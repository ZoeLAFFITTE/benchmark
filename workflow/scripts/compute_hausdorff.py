#!/usr/bin/env python
"""
Compute bidirectional Hausdorff, mean nearest-neighbor, and 95th percentile
distances for each fundus within reproducible basins between the test and
retest spherical registration surfaces of a subject, using Cachia, Trace,
and Mindboggle sulcal extraction methods.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import slam.io as sio
from itertools import combinations
from scipy.spatial import cKDTree


# def hausdorff_distance(set_sense, set_antisense, metric="euclidean"):
#     """
#     Compute bidirectional Hausdorff, mean nearest-neighbor, and 95th percentile
#     distances between two sets of points.

#     Parameters
#     ----------
#     set_sense : numpy.ndarray
#         Coordinates of the points in the first set.
#     set_antisense : numpy.ndarray
#         Coordinates of the points in the second set.
#     metric : str, optional
#         Distance metric used to compute pairwise distances.

#     Returns
#     -------
#     hausdorff : float
#         Bidirectional Hausdorff distance between the two sets of points.
#     mean_distance : float
#         Mean bidirectional nearest-neighbor distance between the two sets.
#     p95_distance : float
#         95th percentile of the bidirectional nearest-neighbor distances.
#     """
#     dists = cdist(set_sense, set_antisense, metric=metric)

#     # Nearest-neighbor distance from A to B.
#     min_distances_sense = np.min(dists, axis=1)

#     # Nearest-neighbor distance from B to A.
#     min_distances_antisense = np.min(dists, axis=0)

#     all_min_distances = np.concatenate([min_distances_sense,
#         min_distances_antisense])

#     hausdorff = np.max(all_min_distances)
#     mean_distance = np.mean(all_min_distances)
#     p95_distance = np.percentile(all_min_distances, 95)

#     return hausdorff, mean_distance, p95_distance

def hausdorff_distance(set_sense, set_antisense):
    """
    Compute bidirectional Hausdorff, mean nearest-neighbor, and 95th percentile
    distances between two sets of points.

    Parameters
    ----------
    set_sense : numpy.ndarray
        Coordinates of the points in the first set.
    set_antisense : numpy.ndarray
        Coordinates of the points in the second set.
    metric : str, optional
        Distance metric used to compute pairwise distances.

    Returns
    -------
    hausdorff : float
        Bidirectional Hausdorff distance between the two sets of points.
    mean_distance : float
        Mean bidirectional nearest-neighbor distance between the two sets.
    p95_distance : float
        95th percentile of the bidirectional nearest-neighbor distances.
    """
    tree_sense = cKDTree(set_sense)
    tree_antisense = cKDTree(set_antisense)

    min_distances_sense = tree_antisense.query(
        set_sense,
        k=1,
    )[0]

    min_distances_antisense = tree_sense.query(
        set_antisense,
        k=1,
    )[0]

    all_min_distances = np.concatenate(
        [min_distances_sense, min_distances_antisense]
    )

    hausdorff = np.max(all_min_distances)
    mean_distance = np.mean(all_min_distances)
    p95_distance = np.percentile(all_min_distances, 95)

    return hausdorff, mean_distance, p95_distance, 


def compute_sulci_hausdorff(
    test_texture,
    retest_texture,
    mesh_test,
    mesh_retest,
    df_overlap,
    subject_id,
    side,
):
    """
    Compute bidirectional distances for fundi within reproducible basins
    between test and retest surfaces.

    Parameters
    ----------
    test_texture : numpy.ndarray
        Sulcal labels on the test surface.
    retest_texture : numpy.ndarray
        Sulcal labels on the retest surface.
    mesh_test : trimesh.Trimesh
        Spherical registration mesh of the test surface.
    mesh_retest : trimesh.Trimesh
        Spherical registration mesh of the retest surface.
    df_overlap : pandas.DataFrame
        DataFrame containing the pairs of reproducible test and retest basin
        labels.
    subject_id : str
        Subject identifier.
    side : str
        Hemisphere.

    Returns
    -------
    pandas.DataFrame
        Test-retest distances for each reproducible basin pair.
    """

    results = []

    for label_test, label_retest in df_overlap[
        ["test_basin", "retest_basin"]
    ].itertuples(index=False, name=None):

        test_mask = test_texture == label_test
        retest_mask = retest_texture == label_retest

        test_points = mesh_test.vertices[test_mask]
        retest_points = mesh_retest.vertices[retest_mask]

        if len(test_points) == 0 or len(retest_points) == 0:
            continue

        else:
            (
                hausdorff,
                mean_nn,
                p95_nn,
            ) = hausdorff_distance(
                test_points,
                retest_points,
            )

        results.append(
            {
                "subject": subject_id,
                "side": side,
                "label_test": int(label_test),
                "label_retest": int(label_retest),
                "hausdorff": hausdorff,
                "mean_nn": mean_nn,
                "p95_nn": p95_nn,
            }
        )

    return pd.DataFrame(results)


def compute_intermethod_hausdorff(
    texture_1,
    texture_2,
    mesh,
    labels,
):
    """
    Compute bidirectional distances between fundi from two methods
    within the same reproducible basins.
    """

    results = []

    for label in labels:

        mask_1 = texture_1 == label
        mask_2 = texture_2 == label

        points_1 = mesh.vertices[mask_1]
        points_2 = mesh.vertices[mask_2]

        if len(points_1) == 0 or len(points_2) == 0:
            continue

        hausdorff, mean_distance, p95_distance = hausdorff_distance(
            points_1,
            points_2,
        )

        results.append(
            {
                "basin_label": int(label),
                "hausdorff": hausdorff,
                "mean_nn": mean_distance,
                "p95_nn": p95_distance,
            }
        )

    return results


df_matches = pd.read_csv(snakemake.input.match_tsv, sep="\t")
df_matches["overlap"] = df_matches["overlap"].astype(float)

df_reproducible = df_matches[df_matches["overlap"] >= snakemake.params.thr_basins]

sphere_reg_test = sio.load_mesh(snakemake.input.sphere_reg_gii_test)
sphere_reg_retest = sio.load_mesh(snakemake.input.sphere_reg_gii_retest)

subject = snakemake.wildcards.sub
side = snakemake.params.side

# -------------------------------------------------------------------------
# Trace
# -------------------------------------------------------------------------
trace_test = sio.load_texture(snakemake.input.sulci_trace_labeled_test).darray[0]
trace_retest = sio.load_texture(snakemake.input.sulci_trace_labeled_retest).darray[0]

trace_results = compute_sulci_hausdorff(
    trace_test,
    trace_retest,
    sphere_reg_test,
    sphere_reg_retest,
    df_reproducible,
    subject,
    side,
)

trace_results.to_csv(snakemake.output.trace_results_tsv, sep="\t", index=False)

print(trace_results)

# -------------------------------------------------------------------------
# Cachia
# -------------------------------------------------------------------------
cachia_test = sio.load_texture(snakemake.input.sulci_cachia_labeled_test).darray[0]
cachia_retest = sio.load_texture(snakemake.input.sulci_cachia_labeled_retest).darray[0]

cachia_results = compute_sulci_hausdorff(
    cachia_test,
    cachia_retest,
    sphere_reg_test,
    sphere_reg_retest,
    df_reproducible,
    subject,
    side,
)

cachia_results.to_csv(snakemake.output.cachia_results_tsv, sep="\t", index=False)

# -------------------------------------------------------------------------
# Mindboggle
# -------------------------------------------------------------------------
mindboggle_test = sio.load_texture(snakemake.input.sulci_mindboggle_labeled_test).darray[0]
mindboggle_retest = sio.load_texture(snakemake.input.sulci_mindboggle_labeled_retest).darray[0]

mindboggle_results = compute_sulci_hausdorff(
    mindboggle_test,
    mindboggle_retest,
    sphere_reg_test,
    sphere_reg_retest,
    df_reproducible,
    subject,
    side,
)

mindboggle_results.to_csv(snakemake.output.mindboggle_results_tsv, sep="\t", index=False)

# -------------------------------------------------------------------------
# Inter-method
# -------------------------------------------------------------------------

methods_test = {
    "Trace": trace_test,
    "Cachia": cachia_test,
    "Mindboggle": mindboggle_test,
}

methods_retest = {
    "Trace": trace_retest,
    "Cachia": cachia_retest,
    "Mindboggle": mindboggle_retest,
}

intermethod_results = []

method_pairs = combinations(
    methods_test.keys(),
    2,
)

# -------------------------------------------------------------------------
# Compare methods within reproducible basins
# -------------------------------------------------------------------------

for method_1, method_2 in method_pairs:

    # ---------------------------------------------------------------------
    # Test
    # ---------------------------------------------------------------------

    labels_test = df_reproducible["test_basin"].unique()

    results = compute_intermethod_hausdorff(
        methods_test[method_1],
        methods_test[method_2],
        sphere_reg_test,
        labels_test,
    )

    for result in results:
        intermethod_results.append(
            {
                "subject": subject,
                "session": "test",
                "side": side,
                "method_1": method_1,
                "method_2": method_2,
                **result,
            }
        )

    # ---------------------------------------------------------------------
    # Retest
    # ---------------------------------------------------------------------

    labels_retest = df_reproducible["retest_basin"].unique()

    results = compute_intermethod_hausdorff(
        methods_retest[method_1],
        methods_retest[method_2],
        sphere_reg_retest,
        labels_retest,
    )

    for result in results:
        intermethod_results.append(
            {
                "subject": subject,
                "session": "retest",
                "side": side,
                "method_1": method_1,
                "method_2": method_2,
                **result,
            }
        )

# -------------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------------

df_intermethod = pd.DataFrame(intermethod_results)

df_intermethod.to_csv(
    snakemake.output.intermethod_results_tsv,
    sep="\t",
    index=False,
)