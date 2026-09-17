import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import slam.io as sio


def hausdorff_distance(setA, setB, metric="euclidean"):
    """Compute Hausdorff, mean nearest-neighbor and 95th percentile distances."""

    dists = cdist(setA, setB)

    # Nearest-neighbor distance from A to B
    min_distances_A = np.min(dists, axis=1)

    # Nearest-neighbor distance from B to A
    min_distances_B = np.min(dists, axis=0)

    all_min_distances = np.concatenate(
        [min_distances_A, min_distances_B]
    )

    hausdorff = np.max(all_min_distances)
    mean_distance = np.mean(all_min_distances)
    p95_distance = np.percentile(all_min_distances, 95)

    return hausdorff, mean_distance, p95_distance


def compute_sulci_hausdorff(
    test_texture,
    retest_texture,
    mesh_test,
    mesh_retest,
    labels_test,
    labels_retest,
):
    """Compute Hausdorff distance for selected basin labels."""

    results_hausdorff = {}
    results_mean = {}
    results_p95 = {}

    for label_test in labels_test:

        # Find corresponding label in retest
        label_retest = labels_retest[labels_test.index(label_test)]

        test_mask = test_texture == label_test
        retest_mask = retest_texture == label_retest

        test_points = mesh_test.vertices[test_mask]
        retest_points = mesh_retest.vertices[retest_mask]

        if len(test_points) == 0 or len(retest_points) == 0:
            results_hausdorff[int(label_test)] = np.nan
            results_mean[int(label_test)] = np.nan
            results_p95[int(label_test)] = np.nan
            continue

        hausdorff, mean_distance, p95_distance = hausdorff_distance(
            test_points,
            retest_points,
        )

        results_hausdorff[int(label_test)] = hausdorff
        results_mean[int(label_test)] = mean_distance
        results_p95[int(label_test)] = p95_distance

    return (
        results_hausdorff,
        results_mean,
        results_p95,
    )


def save_results(results, subject, output_file):
    """Save one subject's results to a TSV file."""

    row = {
        "subject": subject,
        **results,
    }

    df = pd.DataFrame([row])

    df.to_csv(
        output_file,
        sep="\t",
        index=False,
    )


mesh_test = sio.load_mesh(snakemake.input.sphere_reg_gii_test)
mesh_retest = sio.load_mesh(snakemake.input.sphere_reg_gii_retest)

filtered_basins_test = np.load(snakemake.input.filtered_basins_npy_test).tolist()
filtered_basins_retest = np.load(snakemake.input.filtered_basins_npy_retest).tolist()

subject = snakemake.wildcards.sub

print(subject)

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
    mesh_test,
    mesh_retest,
    filtered_basins_test,
    filtered_basins_retest,
)

save_results(trace_results_hausdorff, subject, snakemake.output.trace_results_hausdorff_tsv)
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
    mesh_test,
    mesh_retest,
    filtered_basins_test,
    filtered_basins_retest,
)

save_results(cachia_results_hausdorff, subject, snakemake.output.cachia_results_hausdorff_tsv)
save_results(cachia_results_mean, subject, snakemake.output.cachia_results_mean_tsv)
save_results(cachia_results_p95, subject, snakemake.output.cachia_results_p95_tsv)


# -------------------------------------------------------------------------
# Mindboggle
# -------------------------------------------------------------------------
mindboggle_test = sio.load_texture(snakemake.input.sulci_mindboggle_labeled_test).darray[0]
mindboggle_retest = sio.load_texture(snakemake.input.sulci_mindboggle_labeled_retest).darray[0]

(
    mindboggle_results_hausdorff,
    mindboggle_results_mean,
    mindboggle_results_p95,
) = compute_sulci_hausdorff(
    mindboggle_test,
    mindboggle_retest,
    mesh_test,
    mesh_retest,
    filtered_basins_test,
    filtered_basins_retest,
)

save_results(mindboggle_results_hausdorff, subject, snakemake.output.mindboggle_results_hausdorff_tsv)
save_results(mindboggle_results_mean, subject, snakemake.output.mindboggle_results_mean_tsv)
save_results(mindboggle_results_p95, subject, snakemake.output.mindboggle_results_p95_tsv)