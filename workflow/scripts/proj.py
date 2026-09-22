import numpy as np
import pandas as pd
import slam.io as sio
import slam.plot as splt


METHODS = ["cachia", "trace", "mindboggle"]
METRICS = ["hausdorff", "mean", "p95"]


# -------------------------------------------------------------------------
# Mesh atlas
# -------------------------------------------------------------------------

white_template = sio.load_mesh(
    snakemake.input.white_template_gii
)

n_vertices = len(white_template.vertices)


# -------------------------------------------------------------------------
# Charger les 45 textures de bassins
# -------------------------------------------------------------------------

basins_test_reg = []

for path in snakemake.input.basins_test_reg_gii:

    texture = sio.load_texture(path).darray[0]

    if len(texture) != n_vertices:
        raise ValueError(
            f"{path}: {len(texture)} vertices, "
            f"expected {n_vertices}"
        )

    basins_test_reg.append(texture)


# -------------------------------------------------------------------------
# Charger les TSV
# -------------------------------------------------------------------------

tsv_paths = {
    ("cachia", "hausdorff"):
        snakemake.input.cachia_hausdorff_tsv,

    ("cachia", "mean"):
        snakemake.input.cachia_mean_tsv,

    ("cachia", "p95"):
        snakemake.input.cachia_p95_tsv,

    ("trace", "hausdorff"):
        snakemake.input.trace_hausdorff_tsv,

    ("trace", "mean"):
        snakemake.input.trace_mean_tsv,

    ("trace", "p95"):
        snakemake.input.trace_p95_tsv,

    ("mindboggle", "hausdorff"):
        snakemake.input.mindboggle_hausdorff_tsv,

    ("mindboggle", "mean"):
        snakemake.input.mindboggle_mean_tsv,

    ("mindboggle", "p95"):
        snakemake.input.mindboggle_p95_tsv,
}


data = {}

for key, path in tsv_paths.items():

    df = pd.read_csv(
        path,
        sep="\t",
    )

    df["subject"] = df["subject"].astype(str)

    data[key] = df.set_index("subject")


# -------------------------------------------------------------------------
# Color scale
# -------------------------------------------------------------------------

colorscale = [
    [0.0, "white"],
    [0.1, "blue"],
    [1.0, "red"],
]


# -------------------------------------------------------------------------
# Outputs
# -------------------------------------------------------------------------

outputs = {
    ("cachia", "hausdorff"):
        snakemake.output.cachia_hausdorff,

    ("cachia", "mean"):
        snakemake.output.cachia_mean,

    ("cachia", "p95"):
        snakemake.output.cachia_p95,

    ("trace", "hausdorff"):
        snakemake.output.trace_hausdorff,

    ("trace", "mean"):
        snakemake.output.trace_mean,

    ("trace", "p95"):
        snakemake.output.trace_p95,

    ("mindboggle", "hausdorff"):
        snakemake.output.mindboggle_hausdorff,

    ("mindboggle", "mean"):
        snakemake.output.mindboggle_mean,

    ("mindboggle", "p95"):
        snakemake.output.mindboggle_p95,
}


# -------------------------------------------------------------------------
# 9 plots
# -------------------------------------------------------------------------

for method in METHODS:

    for metric in METRICS:

        metric_data = data[(method, metric)]

        subject_textures = []

        # -------------------------------------------------------------
        # Chaque sujet
        # -------------------------------------------------------------

        for subject_idx, (subject, row) in enumerate(
            metric_data.iterrows()
        ):

            basins = basins_test_reg[subject_idx]

            subject_texture = np.zeros(
                n_vertices,
                dtype=np.float32,
            )

            # ---------------------------------------------------------
            # Chaque bassin
            # ---------------------------------------------------------

            for label, distance in row.items():

                if pd.isna(distance):
                    continue

                label = int(label)

                mask = basins == label

                subject_texture[mask] = distance

            subject_textures.append(subject_texture)


        # -------------------------------------------------------------
        # Moyenne des 45 sujets
        # -------------------------------------------------------------

        mean_texture = np.mean(
            np.stack(subject_textures),
            axis=0,
        )

        # -------------------------------------------------------------
        # Plot
        # -------------------------------------------------------------

        mesh_data = {
            "vertices": white_template.vertices,
            "faces": white_template.faces,
            "title": f"{method.capitalize()} - {metric} - Mean",
        }

        intensity_data = {
            "values": mean_texture,
            "mode": "vertex",
            "cmin": 0,
            "cmax": 30,
            "colorbar": {
                "title": "Distance",
            },
        }

        display_settings = {
            "template": "plotly_white",
            "colorscale": colorscale,
        }

        fig = splt.plot_mesh(
            mesh_data,
            intensity_data,
            display_settings,
            caption=False,
        )

        # -------------------------------------------------------------
        # Sauvegarder
        # -------------------------------------------------------------

        fig.write_html(
            outputs[(method, metric)],
            include_plotlyjs="cdn",
        )