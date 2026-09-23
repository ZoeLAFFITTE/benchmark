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


def hausdorff_distance(set_sense, set_antisense, metric="euclidean"):
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
    dists = cdist(set_sense, set_antisense, metric=metric)

    # Nearest-neighbor distance from A to B.
    min_distances_sense = np.min(dists, axis=1)

    # Nearest-neighbor distance from B to A.
    min_distances_antisense = np.min(dists, axis=0)

    all_min_distances = np.concatenate([min_distances_sense,
        min_distances_antisense])

    hausdorff = np.max(all_min_distances)
    mean_distance = np.mean(all_min_distances)
    p95_distance = np.percentile(all_min_distances, 95)

    return hausdorff, mean_distance, p95_distance


def compute_sulci_hausdorff(
    test_texture,
    retest_texture,
    mesh_test,
    mesh_retest,
    df_overlap,
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

    Returns
    -------
    results_hausdorff : dict
        Bidirectional Hausdorff distance for each test basin.
    results_mean : dict
        Mean bidirectional nearest-neighbor distance for each test basin.
    results_p95 : dict
        95th percentile of the bidirectional nearest-neighbor distances for
        each test basin.
    """
    results_hausdorff = {}
    results_mean = {}
    results_p95 = {}

    for label_test, label_retest in df_overlap[
        ["test_basin", "retest_basin"]
    ].itertuples(index=False, name=None):

        test_mask = test_texture == label_test
        retest_mask = retest_texture == label_retest

        test_points = mesh_test.vertices[test_mask]
        retest_points = mesh_retest.vertices[retest_mask]

        if len(test_points) == 0 or len(retest_points) == 0:
            results_hausdorff[int(label_test)] = np.nan
            results_mean[int(label_test)] = np.nan
            results_p95[int(label_test)] = np.nan
            continue

        bihausdorff, mean_nearest_neighbor, p95_nearest_neighbor = hausdorff_distance(
            test_points,
            retest_points,
        )

        results_hausdorff[int(label_test)] = bihausdorff
        results_mean[int(label_test)] = mean_nearest_neighbor
        results_p95[int(label_test)] = p95_nearest_neighbor

    return (
        results_hausdorff,
        results_mean,
        results_p95,
    )


def save_results(results, subject_id, output_file):
    """
    Save the results for one subject to a TSV file.

    Parameters
    ----------
    results : dict
        Dictionary containing the distance value for each basin.
    subject_id : str
        Subject identifier.
    output_file : str
        Path to the output TSV file.
    """
    row = {
        "subject": subject_id,
        **results,
    }

    df = pd.DataFrame([row])

    df.to_csv(
        output_file,
        sep="\t",
        index=False,
    )


df_matches = pd.read_csv(snakemake.input.match_tsv, sep="\t")
df_matches["overlap"] = df_matches["overlap"].astype(float)

df_reproducible = df_matches[df_matches["overlap"] >= snakemake.params.thr_basins]

sphere_reg_test = sio.load_mesh(snakemake.input.sphere_reg_gii_test)
sphere_reg_retest = sio.load_mesh(snakemake.input.sphere_reg_gii_retest)

subject = snakemake.wildcards.sub

# -------------------------------------------------------------------------
# Trace
# -------------------------------------------------------------------------
trace_test = sio.load_texture(snakemake.input.sulci_trace_labeled_test).darray[0]
trace_retest = sio.load_texture(snakemake.input.sulci_trace_labeled_retest).darray[0]

(
    trace_results_hausdorff,
    trace_results_mean,
    trace_results_p95,
) = compute_sulci_hausdorff(
    trace_test,
    trace_retest,
    sphere_reg_test,
    sphere_reg_retest,
    df_reproducible,
)

save_results(
    trace_results_hausdorff, subject, snakemake.output.trace_results_hausdorff_tsv
)
save_results(trace_results_mean, subject, snakemake.output.trace_results_mean_tsv)
save_results(trace_results_p95, subject, snakemake.output.trace_results_p95_tsv)

# -------------------------------------------------------------------------
# Cachia
# -------------------------------------------------------------------------
cachia_test = sio.load_texture(snakemake.input.sulci_cachia_labeled_test).darray[0]
cachia_retest = sio.load_texture(snakemake.input.sulci_cachia_labeled_retest).darray[0]

(
    cachia_results_hausdorff,
    cachia_results_mean,
    cachia_results_p95,
) = compute_sulci_hausdorff(
    cachia_test,
    cachia_retest,
    sphere_reg_test,
    sphere_reg_retest,
    df_reproducible,
)

save_results(
    cachia_results_hausdorff, subject, snakemake.output.cachia_results_hausdorff_tsv
)
save_results(cachia_results_mean, subject, snakemake.output.cachia_results_mean_tsv)
save_results(cachia_results_p95, subject, snakemake.output.cachia_results_p95_tsv)


# -------------------------------------------------------------------------
# Mindboggle
# -------------------------------------------------------------------------
mindboggle_test = sio.load_texture(
    snakemake.input.sulci_mindboggle_labeled_test
).darray[0]
mindboggle_retest = sio.load_texture(
    snakemake.input.sulci_mindboggle_labeled_retest
).darray[0]

(
    mindboggle_results_hausdorff,
    mindboggle_results_mean,
    mindboggle_results_p95,
) = compute_sulci_hausdorff(
    mindboggle_test,
    mindboggle_retest,
    sphere_reg_test,
    sphere_reg_retest,
    df_reproducible,
)

save_results(
    mindboggle_results_hausdorff,
    subject,
    snakemake.output.mindboggle_results_hausdorff_tsv,
)
save_results(
    mindboggle_results_mean, subject, snakemake.output.mindboggle_results_mean_tsv
)
save_results(
    mindboggle_results_p95, subject, snakemake.output.mindboggle_results_p95_tsv
)
