
#!/usr/bin/env python3
"""
Plot the reference and pruned sulcal skeletons on a cortical surface.

Reference skeleton: black
Pruned skeleton: red
Cortical surface: light grey

The script is designed to run from a Snakemake rule.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import slam.io as sio


def load_mesh(mesh_path):
    """Load a cortical mesh and return its vertices and faces."""
    mesh = sio.load_mesh(str(mesh_path))

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    return vertices, faces


def load_skeleton_vertices(texture_path):
    """Return the indices of vertices belonging to a skeleton texture."""
    texture = sio.load_texture(str(texture_path))
    values = np.asarray(texture.darray).squeeze()

    if values.ndim != 1:
        raise ValueError(
            f"Expected a 1D skeleton texture in {texture_path}, "
            f"but got shape {values.shape}."
        )

    return np.flatnonzero(np.isfinite(values) & (values != 0))


def add_skeleton_trace(fig, vertices, indices, name, color, size=3):
    """Add the skeleton vertices as a 3D scatter trace."""
    if len(indices) == 0:
        print(f"Warning: no skeleton vertices found for '{name}'.")
        return

    fig.add_trace(
        go.Scatter3d(
            x=vertices[indices, 0],
            y=vertices[indices, 1],
            z=vertices[indices, 2],
            mode="markers",
            name=name,
            marker=dict(
                size=size,
                color=color,
                opacity=1.0,
            ),
            hovertemplate=(
                "Vertex index: %{customdata}<extra>" + name + "</extra>"
            ),
            customdata=indices,
        )
    )


def plot_individual_skeletons(
    mesh_path,
    reference_path,
    pruned_path,
    output_path,
    title,
):
    """Plot reference and pruned skeletons on the cortical surface."""
    vertices, faces = load_mesh(mesh_path)

    reference_indices = load_skeleton_vertices(reference_path)
    pruned_indices = load_skeleton_vertices(pruned_path)

    n_vertices = len(vertices)

    for label, indices in (
        ("reference", reference_indices),
        ("pruned", pruned_indices),
    ):
        if len(indices) and indices.max() >= n_vertices:
            raise ValueError(
                f"The {label} texture contains vertex indices outside "
                f"the mesh range [0, {n_vertices - 1}]."
            )

    fig = go.Figure()

    # Cortical surface
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="lightgrey",
            opacity=0.35,
            name="Cortical surface",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # Reference skeleton
    add_skeleton_trace(
        fig,
        vertices,
        reference_indices,
        name="Reference skeleton",
        color="black",
        size=3,
    )

    # Pruned skeleton
    add_skeleton_trace(
        fig,
        vertices,
        pruned_indices,
        name="Pruned skeleton",
        color="red",
        size=4,
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
            ),
        ),
        legend=dict(
            title="Skeleton",
            groupclick="togglegroup",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True)

    print(f"Saved: {output_path}")
    print(f"Reference vertices: {len(reference_indices)}")
    print(f"Pruned vertices: {len(pruned_indices)}")


# ------------------------------------------------------------
# Snakemake entry point
# ------------------------------------------------------------

mesh_path = snakemake.input.mesh
reference_path = snakemake.input.reference
pruned_path = snakemake.input.pruned
output_path = snakemake.output.fig

test = snakemake.wildcards.test
subject = snakemake.wildcards.sub
side = snakemake.wildcards.side
method = snakemake.params.method

if method == "both":
    sdist = snakemake.wildcards.sdist
    slen = snakemake.wildcards.slen
    config_label = f"dist={sdist}, len={slen}"
else:
    seuil = snakemake.wildcards.seuil
    config_label = f"{method} threshold={seuil}"

title = (
    f"{test} | {subject} | {side} | "
    f"{method} | {config_label}"
)

plot_individual_skeletons(
    mesh_path=mesh_path,
    reference_path=reference_path,
    pruned_path=pruned_path,
    output_path=output_path,
    title=title,
)