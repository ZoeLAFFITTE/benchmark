import copy
import networkx as nx
import plotly.express as px
import numpy as np
import slam.io as sio
import slam.texture as stex
import slam.plot as splt
import plotly.graph_objects as go
from collections import Counter
import potpourri3d as pp3d
from scipy.spatial import cKDTree

import pandas as pd
import numpy as np

# Métadonnées issues des wildcards
test = snakemake.wildcards.test
sub  = snakemake.wildcards.sub
side = snakemake.wildcards.side   # hémisphère


def same_cc(active, neigh):
    """
    Teste si les sommets de `active` forment une seule composante connexe
    selon le voisinage `neigh`.

    Retourne True si tous les sommets sont connectés entre eux,
    False sinon.
    """
    active = set(active)

    if not active:
        return True

    # Point de départ
    visited = {next(iter(active))}
    todo = list(visited)

    # Propagation dans la composante connexe
    while todo:
        p = todo.pop()

        for nb in neigh[p]:
            if nb in active and nb not in visited:
                visited.add(nb)
                todo.append(nb)

    return len(visited) == len(active)


def prefilter_vertex(ntex, neigh, v):
    """
    Critère décidant si un vertex peut être érodé.
    Elle retourne une valeur positive si le vertex ``v`` peut être supprimé
    (érodé), ou 0 s'il doit être gelé (conservé).
    Ici : on gèle un sommet s'il n'a qu'UN SEUL voisin actif dans
    l'objet (= une pointe / extrémité de branche), afin de ne pas "manger"
    les extrémités du squelette.

    Paramètres
    ----------
    ntex : numpy.ndarray, shape (N,)
        Tableau numpy de la texture courante (binaire : 0=fond, 1=objet, 2=front).
    neigh : dict
        Dictionnaire de voisinage : ``neigh[v]`` donne les indices des
        vertices voisins de ``v``.
    v : int
        Indice du vertex à évaluer.

    Retour
    ------
        ``0`` si le vertex est gelé (ne peut pas être supprimé),
        ``1`` s'il peut être érodé.
        :rtype: int
    """

    # Liste des indices des voisins directs de v
    n_v = np.array(list(neigh[v]))

    # Valeurs de la texture (0, 1 ou 2) pour chacun de ces voisins
    nval = ntex[n_v]

    # Sous-ensemble des voisins qui sont "actifs"
    active = n_v[nval != 0]

    # Si le vertex n'a pas au moins 2 voisins actifs, on le gèle
    if len(active) < 2:
        return 0

    # Sinon (2 voisins actifs ou plus), on autorise l'érosion
    return 1


def filter_vertex(ntex, v, neigh, c):
    """
    Filter a vertex while preserving the local connectivity of the object.
    """
    if c <= 0:
        return None

    active = np.array(list(neigh[v]))
    active = active[ntex[active] != 0]

    if len(active) == 0:
        return None

    if len(active) == 1 or same_cc(active, neigh) == 1:
        return active

    return None


def sort_potential(front, texture):
    """
    Trie les vertices du front selon la valeur de la texture
    de potentiel, par ordre croissant.
    """

    texture = np.asarray(texture)

    if texture.ndim > 1:
        texture = texture[0]

    return sorted(front, key=lambda v: texture[v])


def extract_borders(indices, neighbor_lists):
    indices_set = set(indices)

    borders = [
        v for v in indices
        if any(n not in indices_set for n in neighbor_lists[v])
    ]

    return borders


def mesh_skeleton(neigh, indices, dist_tex, do_timesteps=False, verbose=False):
    """
    Calcule le squelette d'un objet défini par une texture binaire
    sur une zone du maillage définie par `indices`.

    L'érosion est asymétrique : à chaque étape, le sommet du front
    ayant la plus petite valeur de `dist_tex` est traité en priorité.

    Parameters
    ----------
    neigh : dict
        Dictionnaire de voisinage des vertices.
    indices : array-like
        Indices globaux des vertices appartenant à l'objet.
    dist_tex : array-like
        Carte de profondeur utilisée pour prioriser l'érosion.
        Les valeurs les plus faibles sont traitées en premier.
    do_timesteps : bool, optional
        Si True, retourne les textures des différentes étapes.
    verbose : bool, optional
        Si True, affiche des informations sur chaque phase.

    Returns
    -------
    out_tex : list ou ndarray
        Textures des étapes si `do_timesteps=True`,
        sinon texture finale.
    moments : list
        Textures décrivant les événements de chaque phase.
    curv_moments : list
        Textures décrivant le critère de filtrage des fronts.
    """

    # INITIALISATION ---------------------------------------------------------
    indices = np.asarray(indices, dtype=int)
    dist_tex = np.asarray(dist_tex)

    ntex = np.zeros(len(dist_tex), dtype=np.int16)
    ntex[indices] = 1

    if verbose:
        print(f"Vertices du fold en entrée : {len(indices)}")

    # FRONT INITIAL ----------------------------------------------------------
    front = set(extract_borders(indices, neigh))

    # TEXTURES DE SORTIE -----------------------------------------------------
    out_tex = []
    moments = []
    curv_moments = []

    frozen = False
    p = 0

    # BOUCLE PRINCIPALE ------------------------------------------------------
    while not frozen:

        frozen_points = []
        eroded_points = []
        new_front_points = []

        frozen = True

        if verbose:
            print(f"iter {p}: front: {len(front)}")

        # SNAPSHOT -----------------------------------------------------------
        if do_timesteps:
            timestep = np.zeros(ntex.shape, dtype=np.int16)
            timestep[ntex != 0] = 2

            if front:
                timestep[list(front)] = 1

            out_tex.append(timestep)

        # MARQUAGE DU FRONT --------------------------------------------------
        if front:
            ntex[list(front)] = 2

        # PRÉ-FILTRAGE DU FRONT ---------------------------------------------
        nfrontbis = np.zeros(ntex.shape, dtype=np.int16)
        nfrontbis[ntex != 0] = 2

        curv_rejected = []
        curv_accepted = []

        for v in front:
            c = prefilter_vertex(ntex, neigh, v)

            if c <= 0:
                curv_rejected.append(v)
            else:
                curv_accepted.append(v)

        if curv_rejected:
            nfrontbis[curv_rejected] = 3

        if curv_accepted:
            nfrontbis[curv_accepted] = 1

        curv_moments.append(nfrontbis)

        # ÉROSION ------------------------------------------------------------
        new_front = []
        new_front_set = set()
        new_front_points = []

        while front:

            # Sélection du sommet de profondeur minimale
            v = min(front, key=lambda x: dist_tex[x])
            front.remove(v)

            # Recalcul du critère dans l'état courant du maillage
            c = prefilter_vertex(ntex, neigh, v)
            v2 = filter_vertex(ntex, v, neigh, c)

            # Le sommet ne peut pas être érodé
            if v2 is None:
                new_front.append(v)
                new_front_set.add(v)
                frozen_points.append(v)
                continue

            # Érosion du sommet
            ntex[v] = 0
            eroded_points.append(v)
            frozen = False

            # Sélection des nouveaux sommets du front
            v2 = [
                x for x in v2
                if x not in front and x not in new_front_set
            ]

            # Mise à jour du nouveau front
            new_front.extend(v2)
            new_front_set.update(v2)
            new_front_points.extend(v2)

            # Les nouveaux sommets peuvent être traités immédiatement
            if v2:
                ntex[v2] = 2
                front.update(v2)

        # TEXTURE DES ÉVÉNEMENTS DE LA PHASE -------------------------------
        nfront = np.zeros(ntex.shape, dtype=np.int16)
        nfront[ntex != 0] = 2

        if frozen_points:
            nfront[frozen_points] = 3

        if eroded_points:
            nfront[eroded_points] = 4

        if new_front_points:
            nfront[new_front_points] = 1

        moments.append(nfront)

        # PHASE SUIVANTE -----------------------------------------------------
        front = set(new_front)
        p += 1

    # TEXTURE FINALE ---------------------------------------------------------
    final_tex = np.zeros(ntex.shape, dtype=np.int16)
    final_tex[ntex != 0] = 2

    if front:
        final_tex[list(front)] = 1

    if do_timesteps:
        out_tex.append(final_tex)
    else:
        out_tex = final_tex

    return out_tex, moments, curv_moments


def get_ordered_boundary(basin_vertices, white_mesh):
    """Vérifie si le contour du bassin forme une boucle simple."""

    basin_vertices = np.asarray(basin_vertices, dtype=int)
    faces = np.asarray(white_mesh.faces)

    # Faces entièrement incluses dans le bassin
    basin_faces = faces[np.isin(faces, basin_vertices).all(axis=1)]

    if len(basin_faces) == 0:
        print("⚠️ Bordure NON SIMPLE — aucune face dans le bassin.")
        return True

    # Toutes les arêtes des faces retenues
    edges = np.concatenate(
        [
            basin_faces[:, [0, 1]],
            basin_faces[:, [1, 2]],
            basin_faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges.sort(axis=1)

    # Comptage vectorisé des arêtes
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    # Graphe du contour
    G = nx.Graph()
    G.add_edges_from(boundary_edges)

    degrees = [d for _, d in G.degree()]
    n_components = nx.number_connected_components(G)

    simple_loop = (
        G.number_of_nodes() > 0
        and n_components == 1
        and all(d == 2 for d in degrees)
        and G.number_of_edges() == G.number_of_nodes()
    )

    if simple_loop:
        print("✅ Boucle simple.")
        return False

    print(
        f"⚠️ Bordure NON SIMPLE — sommets : {G.number_of_nodes()}, "
        f"arêtes : {G.number_of_edges()}, "
        f"composantes : {n_components}, "
        f"degrés : {sorted(set(degrees))}"
    )
    return True


def _get_frozen_borders(borders, ntex, neigh):
    """Retourne les sommets de bordure gelés selon les critères d'érosion."""

    frozen = []

    for v in borders:
        criteria = prefilter_vertex(ntex, neigh, v)

        if filter_vertex(ntex, v, neigh, criteria) is None:
            frozen.append(v)

    return frozen


def control_borders(indices_fold, borders, dpf_star, neigh):
    """Vérifie si la bordure contient des sommets gelés."""

    ntex = np.zeros(len(dpf_star), dtype=np.int16)
    ntex[indices_fold] = 1

    frozen = _get_frozen_borders(borders, ntex, neigh)

    if not frozen:
        print(
            "✅ Aucun candidat au gel : "
            "toute la bordure est érodable dès le départ."
        )
        return False

    print(f"⚠️ {len(frozen)} point(s) à GELER dès le départ")

    # Distinguer les sommets isolés des extrémités
    isolated, tips = [], []

    for v in frozen:
        active = np.asarray(list(neigh[v]), dtype=int)
        active = active[ntex[active] != 0]

        if len(active) == 0:
            isolated.append(v)
        else:
            tips.append(v)

    if isolated:
        print(f"   - {len(isolated)} point(s) ISOLÉ(S)")
    if tips:
        print(f"   - {len(tips)} extrémité(s) de branche")

    return True


def prune_frozen_borders(indices_fold, neigh, dpf_star):
    """
    Supprime itérativement les sommets de bordure gelés.
    La texture binaire est mise à jour uniquement pour les sommets
    supprimés.
    """

    basin = set(np.asarray(indices_fold, dtype=int))
    ntex = np.zeros(len(dpf_star), dtype=np.int16)
    ntex[list(basin)] = 1

    n_removed = 0

    while basin:
        borders = extract_borders(np.fromiter(basin, dtype=int), neigh)
        frozen = _get_frozen_borders(borders, ntex, neigh)

        if not frozen:
            break

        # Retirer les sommets gelés du bassin et de la texture
        basin.difference_update(frozen)
        ntex[frozen] = 0
        n_removed += len(frozen)

    if n_removed:
        print(f"⚠️ {n_removed} sommet(s) gelé(s) retiré(s) du bassin.")
    else:
        print("✅ Aucun point gelé.")

    return np.asarray(sorted(basin), dtype=int)


def filter_skeleton_dijkstra(white_mesh, skeleton, dpfstar_reversed, indices_fold):
    """
    Filtre le squelette en conservant l'union des plus courts chemins
    entre le pit sulcal et les extrémités accessibles.

    Returns
    -------
    filtered_skeleton : ndarray
        Sommets du squelette parcourus par au moins un chemin retenu.
    endpoints : list of int
        Extrémités du squelette initial accessibles depuis le pit.
    bifurcations : list of int
        Sommets où les chemins empruntent plusieurs successeurs.
    pit : int
        Indice du pit sulcal.
    paths : list of list
        Plus courts chemins du pit vers les extrémités accessibles.
    """

    skeleton = np.asarray(skeleton, dtype=int)
    indices_fold = np.asarray(indices_fold, dtype=int)
    dpfstar_reversed = np.asarray(dpfstar_reversed, dtype=float)

    # Garde : squelette vide ou d'un seul sommet -> pas de chemin possible
    if skeleton.size <= 1:
        print("⚠️ Squelette vide ou d'un seul sommet : fold ignoré.")
        pit = int(skeleton[0]) if skeleton.size == 1 else None
        return skeleton, None, None, pit, None

  
    vals = dpfstar_reversed[indices_fold]
    vmax = vals.max()
    dpf_shifted = vmax - dpfstar_reversed + 1e-6

    # Extraire les arêtes dont les deux sommets appartiennent au squelette
    edges = white_mesh.edges_unique
    lengths = white_mesh.edges_unique_length

    in_skeleton = np.zeros(len(white_mesh.vertices), dtype=bool)
    in_skeleton[skeleton] = True

    mask = in_skeleton[edges[:, 0]] & in_skeleton[edges[:, 1]]
    skeleton_edges = edges[mask]
    lengths = lengths[mask]

    # Pondérer les arêtes selon leur longueur et les valeurs de DPF*
    dpf_cost = dpf_shifted[skeleton_edges].mean(axis=1)
    weights = lengths * dpf_cost

    if np.any(weights <= 0):
        raise ValueError("Les poids des arêtes doivent être positifs.")

    # Construire le graphe pondéré
    graph = nx.Graph()
    graph.add_nodes_from(skeleton)
    graph.add_weighted_edges_from(
        (int(a), int(b), float(w))
        for (a, b), w in zip(skeleton_edges, weights)
    )

    # Identifier le pit sulcal : dans le bassin si possible, sinon dans le squelette
    if indices_fold.size > 0:
        pit = int(indices_fold[np.argmax(dpfstar_reversed[indices_fold])])
    else:
        pit = int(skeleton[np.argmax(dpfstar_reversed[skeleton])])
        print("Bassin vide : pit pris sur le squelette.")

    # Le pit doit être un sommet du graphe du squelette
    if pit not in graph:
        pit = int(skeleton[np.argmax(dpfstar_reversed[skeleton])])
        print("Le pit sulcal n'appartient pas au squelette.")

    # Identifier les extrémités du squelette initial
    endpoints = [
        int(v) for v, degree in graph.degree()
        if degree == 1 and v != pit
    ]

    # Calculer les plus courts chemins en un seul Dijkstra
    all_paths = nx.single_source_dijkstra_path(graph, pit, weight="weight")
    endpoints = [v for v in endpoints if v in all_paths]
    paths = [all_paths[v] for v in endpoints]

    # Conserver l'union des sommets parcourus par les chemins
    filtered_skeleton = (
        np.unique(np.concatenate(paths))
        if paths else np.array([pit], dtype=int)
    )

    # Détecter les bifurcations à partir des chemins retenus
    successors = {}

    for path in paths:
        for a, b in zip(path[:-1], path[1:]):
            successors.setdefault(a, set()).add(b)

    bifurcations = [
        int(v) for v, next_vertices in successors.items()
        if len(next_vertices) >= 2
    ]

    return filtered_skeleton, endpoints, bifurcations, pit, paths


def probability_paths(white_mesh, paths):
    """
    Calcule et affiche la carte de probabilité des chemins.

    Parameters
    ----------
    white_mesh : trimesh.Trimesh
        Maillage cortical.
    paths : list of list
        Plus courts chemins du pit vers les extrémités.

    Returns
    -------
    probability_map : ndarray
        Nombre de chemins passant par chaque sommet.
    """
    vertices = np.asarray(white_mesh.vertices)

    # Compter le nombre de passages sur chaque sommet
    probability_map = np.zeros(len(vertices), dtype=int)

    for path in paths:
        probability_map[path] += 1

    return probability_map


def segment_mesh(
    white_mesh,
    filtered_skeleton,
    pit,
    bifurcations,
    endpoints,
    paths,
    dpfstar_reversed,
):
    """Décompose le squelette en segments et calcule leurs caractéristiques.

    Parameters
    ----------
    white_mesh : slam.mesh.Mesh
        Maillage cortical.
    filtered_skeleton : array-like
        Indices des sommets du squelette filtré.
    pit : int
        Indice du pit sulcal.
    bifurcations : array-like
        Indices des sommets de bifurcation.
    endpoints : array-like
        Indices des extrémités du squelette.
    paths : list of array-like
        Chemins entre le pit et les extrémités.
    dpfstar_reversed : ndarray
        Texture DPF* indexée par sommet.

    Returns
    -------
    segments : list of dict
        Segments et leurs caractéristiques, notamment leur nombre de
        passages complets dans les chemins fournis.
    graph : networkx.Graph
        Graphe du squelette filtré.
    """

    filtered_skeleton = np.asarray(filtered_skeleton, dtype=int)
    dpfstar_reversed = np.asarray(dpfstar_reversed)
    bifurcations = set(map(int, bifurcations))
    endpoints = set(map(int, endpoints))
    pit = int(pit)

    # 1. Construire le graphe à partir des chemins
    graph = nx.Graph()
    graph.add_nodes_from(filtered_skeleton)

    edges = {
        tuple(sorted((int(u), int(v))))
        for path in paths
        for u, v in zip(path[:-1], path[1:])
    }

    # Longueur géométrique de chaque arête
    vertices = np.asarray(white_mesh.vertices)

    edges_array = np.asarray(list(edges), dtype=int)

    if len(edges_array):
        lengths = np.linalg.norm(
            vertices[edges_array[:, 0]] - vertices[edges_array[:, 1]],
            axis=1,
        )
    else:
        lengths = np.array([], dtype=float)

    graph.add_edges_from(
        (int(u), int(v), {"length": float(length)})
        for (u, v), length in zip(edges_array, lengths)
    )

    # 2. Identifier les sommets délimitant les segments
    key_nodes = bifurcations | endpoints | {pit}
    key_nodes.update(
        int(vertex)
        for vertex, degree in graph.degree()
        if degree != 2
    )

    # 3. Décomposer le graphe en segments
    visited_edges = set()
    segments = []

    for start in key_nodes:
        if start not in graph:
            continue

        for neighbor in graph.neighbors(start):
            first_edge = (min(start, neighbor), max(start, neighbor))

            if first_edge in visited_edges:
                continue

            segment_vertices = [start]
            previous, current = start, neighbor
            visited_edges.add(first_edge)

            while True:
                segment_vertices.append(current)

                if current in key_nodes:
                    break

                next_nodes = [
                    vertex
                    for vertex in graph.neighbors(current)
                    if vertex != previous
                ]

                if len(next_nodes) != 1:
                    break

                next_vertex = next_nodes[0]
                edge = (
                    min(current, next_vertex),
                    max(current, next_vertex),
                )

                if edge in visited_edges:
                    break

                visited_edges.add(edge)
                previous, current = current, next_vertex

            segment_edges = frozenset(
                (min(u, v), max(u, v))
                for u, v in zip(
                    segment_vertices[:-1],
                    segment_vertices[1:],
                )
            )

            segment_length = sum(
                graph[u][v]["length"]
                for u, v in zip(
                    segment_vertices[:-1],
                    segment_vertices[1:],
                )
            )

            segments.append({
                "start": int(segment_vertices[0]),
                "end": int(segment_vertices[-1]),
                "vertices": segment_vertices,
                "edges": segment_edges,
                "length": float(segment_length),
                "dpf_mean": float(
                    np.mean(dpfstar_reversed[segment_vertices])
                ),
                "passages": 0,
            })

    # 4. Calculer le nombre de chemins traversant intégralement chaque segment
    path_edge_sets = []

    for path in paths:
        path = np.asarray(path, dtype=int)

        path_edges = frozenset(
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in zip(path[:-1], path[1:])
        )

        path_edge_sets.append(path_edges)

    for segment in segments:
        segment_edges = segment["edges"]

        segment["passages"] = sum(
            segment_edges.issubset(path_edges)
            for path_edges in path_edge_sets
        )

    if not segments:
        print("Aucun segment à analyser.")

    return segments, graph


def basin_stats(basin_id, skeleton, pruned_graph, pit_expected, dpfstar_reversed):
    """
    Calcule les stats du squelette d'un bassin (une ligne de DataFrame).

    Parameters
    ----------
    basin_id : int
        Label du bassin.
    skeleton : array-like
        Sommets du squelette issu de l'érosion (avant Dijkstra).
    pruned_graph : networkx.Graph ou None
        Graphe final (après Dijkstra + pruning). None dans le cas dégénéré.
    pit_expected : int ou None
        Sommet de DPF* maximal dans le fold.
    dpfstar_reversed : ndarray
        Texture DPF*.
    """
    skeleton = np.asarray(skeleton, dtype=int)

    # Le pit (max DPF* du fold) est-il dans le squelette ?
    pit_true = (
        pit_expected is not None and int(pit_expected) in set(skeleton.tolist())
    )

    # Cas dégénéré : squelette vide ou d'un seul sommet
    if pruned_graph is None:
        return {
            "test": test, "sub": sub, "hemisphere": side,
            "basin": int(basin_id),
            "n_vertices": int(skeleton.size),
            "length": 0.0,
            "n_endpoints": 0,
            "n_bifurcations": 0,
            "n_segments": 0,
            "n_pit_true": int(pit_true),
            "n_pit_false": int(not pit_true),
        }

    degree = dict(pruned_graph.degree())

    n_endpoints = sum(d == 1 for d in degree.values())
    n_bifurcations = sum(d >= 3 for d in degree.values())

    # Segments = arêtes du graphe contracté entre nœuds clés
    # (extrémités, bifurcations, pit) ; dans un arbre : somme des degrés / 2
    key_nodes = {v for v, d in degree.items() if d != 2}
    if pit_expected is not None and int(pit_expected) in pruned_graph:
        key_nodes.add(int(pit_expected))
    n_segments = sum(degree[v] for v in key_nodes) // 2

    # Longueur totale du squelette élagué (mm)
    length = pruned_graph.size(weight="length")

    return {
        "test": test,
        "sub": sub,
        "hemisphere": side,
        "basin": int(basin_id),
        "n_vertices": pruned_graph.number_of_nodes(),
        "length": float(length),
        "n_endpoints": int(n_endpoints),
        "n_bifurcations": int(n_bifurcations),
        "n_segments": int(n_segments),
        "n_pit_true": int(pit_true),        # le pit max(DPF*) est dans le squelette
        "n_pit_false": int(not pit_true),   # le point le plus profond du squelette n'est pas le pit
    }


def pruning(
    graph, bifurcations, paths, endpoints, white_mesh, pit,
    min_dist=0, min_length=0,
):
    """Remove short terminal branches in two stages."""

    dist = min_dist * np.cbrt(white_mesh.volume)
    length_b = min_length * np.cbrt(white_mesh.volume)

    vertices = np.asarray(white_mesh.vertices)
    endpoints = list(endpoints)
    bifurcations = set(bifurcations)
    paths = dict(zip(endpoints, paths))
    pruned_graph = graph.copy()
    alive = set(endpoints)

    # Cache edge lengths.
    edge_lengths = {}

    def edge_length(u, v):
        key = (min(u, v), max(u, v))
        if key not in edge_lengths:
            edge_lengths[key] = np.linalg.norm(vertices[u] - vertices[v])
        return edge_lengths[key]

    def branch_length(branch):
        return sum(edge_length(u, v) for u, v in zip(branch, branch[1:]))

    def terminal_branch(endpoint):
        if endpoint not in pruned_graph or pruned_graph.degree(endpoint) != 1:
            return [], None

        branch, previous, current = [], None, endpoint

        while pruned_graph.degree(current) < 3:
            branch.append(current)
            neighbors = [
                n for n in pruned_graph.neighbors(current) if n != previous
            ]
            if len(neighbors) != 1:
                return [], None
            previous, current = current, neighbors[0]

        return branch, current

    def remove_branch(endpoint, branch, stage, bif, length):
        print(
            f"[Pruning {stage}] Endpoint: {endpoint} | "
            f"Bifurcation: {bif} | Longueur supprimée: {length:.2f} mm"
        )
        pruned_graph.remove_nodes_from(branch)
        alive.discard(endpoint)

    # Stage 1: compare nearby endpoints sharing a bifurcation.
    path_bifs = {
        e: set(paths[e]) & bifurcations for e in endpoints
    }
    endpoint_array = np.asarray(endpoints, dtype=int)

    pairs = (
        cKDTree(vertices[endpoint_array]).query_pairs(
            dist, output_type="ndarray"
        )
        if len(endpoints) > 1
        else np.empty((0, 2), dtype=int)
    )
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)

    if len(pairs):
        solver = pp3d.MeshHeatMethodDistanceSolver(
            vertices, white_mesh.faces
        )
        by_source = {}
        for i, j in pairs:
            by_source.setdefault(i, []).append(j)

        close_pairs = []
        for i, js in by_source.items():
            distances = solver.compute_distance(int(endpoint_array[i]))
            close_pairs.extend(
                (distances[endpoint_array[j]], i, j)
                for j in js
                if distances[endpoint_array[j]] < dist
            )
        close_pairs.sort(key=lambda x: x[0])

        for _, i, j in close_pairs:
            a, b = endpoints[i], endpoints[j]
            if a not in alive or b not in alive:
                continue

            common = path_bifs[a] & path_bifs[b]
            if not common:
                continue

            bif = next((n for n in reversed(paths[a]) if n in common), None)
            if bif is None:
                continue

            branch_a, bif_a = terminal_branch(a)
            branch_b, bif_b = terminal_branch(b)
            if not branch_a or not branch_b or bif_a != bif_b:
                continue

            len_a, len_b = branch_length(branch_a), branch_length(branch_b)
            endpoint, branch, length = (
                (a, branch_a, len_a)
                if len_a <= len_b
                else (b, branch_b, len_b)
            )
            remove_branch(endpoint, branch, 1, bif, length)

    # Stage 2: iteratively remove all short terminal branches.
    removed = 0
    while True:
        candidates = []
        for endpoint, degree in pruned_graph.degree():
            if degree != 1 or endpoint == pit:
                continue
            branch, bif = terminal_branch(endpoint)
            if branch and bif is not None:
                length = branch_length(branch)
                if length <= length_b:
                    candidates.append(endpoint)

        if not candidates:
            break

        changed = False
        for endpoint in candidates:
            if endpoint == pit:
                continue
            branch, bif = terminal_branch(endpoint)
            if not branch or bif is None:
                continue

            length = branch_length(branch)
            if length <= length_b:
                remove_branch(endpoint, branch, 2, bif, length)
                removed += 1
                changed = True

        if not changed:
            break

    print(
        f"Deuxième pruning terminé : {removed} branches supprimées "
        f"(seuil <= {length_b} mm)."
    )
    return pruned_graph


if __name__ == "__main__":

    seuil_dist = getattr(snakemake.params, "seuil_dist", 0)
    seuil_len = getattr(snakemake.params, "seuil_len", 0)

    #paths
    white_path = snakemake.input.white_gii
    dpf_white_path = snakemake.input.dpf_white_gii
    basins_path = snakemake.input.basins_white_gii

    #chargement des données
    white_mesh = sio.load_mesh(white_path)
    dpf_star = sio.load_texture(dpf_white_path).darray[0]
    basins = sio.load_texture(basins_path).darray[0]

    #séléction des bassins reproductibles
    labels_basins = np.unique(basins)
    print(labels_basins)

    vertices_dic = {}
    for label in labels_basins:
        vertices = np.where(basins == label)[0]
        vertices_dic[label] = vertices

    #res finaux
    all_basins = np.zeros(len(dpf_star), dtype=bool)
    all_borders = np.zeros(len(dpf_star), dtype=bool)
    all_skeletons = np.zeros(len(dpf_star), dtype=bool)
    all_paths = np.zeros(len(dpf_star), dtype=int)
    global_probability = np.zeros(len(white_mesh.vertices), dtype=int)
    all_pits = []
    all_bifurcations = []
    all_endpoints = []

    dpfstar_reversed = dpf_star #[-x for x in dpf_star]
    dpfstar_reversed = np.asarray(dpfstar_reversed, dtype=float)

    #voisnage
    neigh = white_mesh.vertex_neighbors
    # stats_rows = []

    # ------------------------------------------------------------------
    # ALGORITHME PRINCIPAL
    # ------------------------------------------------------------------
    for label in labels_basins:
        indices_fold = vertices_dic[label]
        print("\n" + "=" * 50)
        print(f"BASIN: {label}")
        print("=" * 50)

        gyri = extract_borders(indices_fold, neigh)
        all_basins[gyri] = True

        #calcul du seuil dpf
        dpf_fold = dpfstar_reversed[indices_fold]
        threshold_dpf = np.percentile(dpf_fold, 25)
        print(f"SEUIL DPF DU BASSIN:{threshold_dpf}")

        indices_fold = indices_fold[
            dpfstar_reversed[indices_fold] >= threshold_dpf
        ]

        # Bordure initiale
        borders = extract_borders(indices_fold, neigh)

        print("\n" + "-" * 50)
        print("SELECTION DU FOLD : bordure initiale du bassin")
        print("-" * 50)

        new_indices_fold = indices_fold
        new_borders = borders

        border_problem = get_ordered_boundary(indices_fold, white_mesh)

        # 1. Corriger la géométrie par seuillage
        if border_problem:
            print("\n" + "-" * 50)
            print("SELECTION DU FOLD : choix d'un nouveau seuil")
            print("-" * 50)

            for q in np.linspace(0.25, 0.75, 50):
                threshold_dpf = np.quantile(dpf_fold, q)

                candidate = indices_fold[
                    dpfstar_reversed[indices_fold] >= threshold_dpf
                ]

                print(f"q={q:.3f} | seuil={threshold_dpf:.6f}")

                if not get_ordered_boundary(candidate, white_mesh):
                    new_indices_fold = candidate
                    break

            print(f"NOUVEAU SEUIL : {threshold_dpf:.6f}")

            new_borders = extract_borders(new_indices_fold, neigh)

        # 2. Supprimer les sommets gelés
        print("\n" + "-" * 50)
        print("SELECTION DU FOLD : correction des bordures gelées")
        print("-" * 50)

        while control_borders(
            new_indices_fold, new_borders, dpfstar_reversed, neigh
        ):
            new_indices_fold = prune_frozen_borders(
                new_indices_fold, neigh, dpfstar_reversed
            )
            new_borders = extract_borders(new_indices_fold, neigh)
        
        # Bassin entièrement érodé par prune_frozen_borders : on l'ignore
        if len(new_indices_fold) == 0:
            print(f"⚠️ Bassin {label} vide après suppression des sommets gelés : ignoré.")
            # stats_rows.append(
            #     basin_stats(label, np.array([], dtype=int), None, None, dpfstar_reversed)
            # )
            print(f"FIN BASIN: {label} (bassin vide)")
            continue

        all_borders[new_borders] = True

        # SQUELETTE
        print("\n" + "-" * 50)
        print("ALGO PRINCIPAL : érosion")
        print("-" * 50)
    
        stex_front, moments, curv_moments = mesh_skeleton(
            neigh, new_indices_fold, dpfstar_reversed,
        )
        skeleton = np.where(stex_front != 0)[0]

        print("\n" + "-" * 50)
        print("ALGO PRINCIPAL : dijkstra")
        print("-" * 50)
        pit_expected = int(new_indices_fold[np.argmax(dpfstar_reversed[new_indices_fold])])
        filtered_skeleton, endpoints, bifurcations, pit, paths = filter_skeleton_dijkstra(
            white_mesh,
            skeleton,
            dpfstar_reversed,
            new_indices_fold,
        )
        if endpoints is None:
            # Cas dégénéré : pas de probability_map ni de pruning.
            # all_basins et all_borders sont déjà renseignés plus haut dans la boucle.
            all_skeletons[skeleton] = True      # 0 ou 1 sommet
            if pit is not None:
                all_pits.append(int(pit))       # all_paths[pit] = 100 à la fin
            print(f"FIN BASIN: {label} (cas dégénéré)")
            # stats_rows.append(
            #     basin_stats(label, skeleton, None, pit_expected, dpfstar_reversed)
            # )
            continue

        probability_map = probability_paths(white_mesh, paths)

        print("\n" + "-" * 50)
        print("PRUNING : segmentation")
        print("-" * 50)
        segments, graph = segment_mesh(
            white_mesh,
            filtered_skeleton,
            pit,
            bifurcations,
            endpoints,
            paths,
            dpfstar_reversed,
        )

        print("\n" + "-" * 50)
        print("PRUNING : filtrage")
        print("-" * 50)
        pruned_graph = graph

        if seuil_dist !=0 or seuil_len!= 0:
            pruned_graph = pruning(graph, bifurcations, paths, endpoints, white_mesh, pit, min_dist=seuil_dist, min_length=seuil_len)

        print("\n" + "-" * 50)
        print(f"FIN BASIN: {label}")
        print("-" * 50)
        all_skeletons[list(pruned_graph.nodes)] = True

        # stats_rows.append(
        #     basin_stats(label, skeleton, pruned_graph, pit_expected, dpfstar_reversed)
        # )

        # Accumulation des résultats de ce basin (les basins sont disjoints)
        mask = probability_map > 0
        global_probability[mask] = probability_map[mask]
        all_pits.append(int(pit))
        all_bifurcations.extend(int(v) for v in bifurcations)
        all_endpoints.extend(int(v) for v in endpoints)
            

    print("\n" + "-" * 50)
    print(f"FIN: Sauvegarde des textures")
    print("-" * 50)
    # Texture finale des chemins (une seule fois, après tous les basins)
    all_paths[global_probability > 0] = global_probability[global_probability > 0]
    all_paths[all_pits] = 100
    all_paths[all_bifurcations] = 200
    all_paths[all_endpoints] = 300

    print("\n" + "-" * 50)
    print("FIN: Sauvegarde des textures")
    print("-" * 50)

    sio.write_texture(stex.TextureND(darray=all_skeletons), snakemake.output.skeletons)
    # sio.write_texture(stex.TextureND(darray=all_borders), snakemake.output.borders)
    # sio.write_texture(stex.TextureND(darray=all_basins), snakemake.output.basins)
    # sio.write_texture(stex.TextureND(darray=all_paths), snakemake.output.proba)

    # df_stats = pd.DataFrame(stats_rows)
    # df_stats.to_csv(snakemake.output.stats, sep="\t", index=False)

    # # ------------------------------------------------------------------
    # # Figure : tous les basins
    # # ------------------------------------------------------------------
    # fig = splt.plot_mesh(
    #     {"vertices": white_mesh.vertices, "faces": white_mesh.faces},
    #     {"values": dpfstar_reversed, "mode": "vertex"},
    #     {"template": "plotly_white", "colorscale": "Sunset"},
    #     caption=False,
    # )

    # mesh_vertices = np.asarray(white_mesh.vertices)
    # mesh_normals = np.asarray(white_mesh.vertex_normals)

    # def lifted(indices, offset):
    #     indices = np.asarray(indices, dtype=int)
    #     return mesh_vertices[indices] + offset * mesh_normals[indices]

    # points = lifted(np.arange(len(mesh_vertices)), 0.3)

    # # Sommets selon leur nombre de passages (tous basins confondus)
    # counts = np.unique(global_probability[global_probability > 0])
    # colors = px.colors.sample_colorscale(
    #     "Turbo", np.linspace(0.05, 0.95, max(len(counts), 2))
    # )

    # for count, color in zip(counts, colors):
    #     indices = np.flatnonzero(global_probability == count)
    #     fig.add_trace(
    #         go.Scatter3d(
    #             x=points[indices, 0],
    #             y=points[indices, 1],
    #             z=points[indices, 2],
    #             mode="markers",
    #             marker={"size": 3, "color": color},
    #             name=f"{count} passage{'s' if count > 1 else ''}",
    #             text=[f"vertex {v}<br>{count} passage(s)" for v in indices],
    #             hoverinfo="text",
    #         )
    #     )

    # # Bifurcations, extrémités et pits de tous les basins
    # if all_bifurcations:
    #     fig.add_trace(
    #         splt.plot_points(
    #             lifted(all_bifurcations, 0.4),
    #             marker={"size": 5, "color": "grey", "symbol": "diamond"},
    #         )
    #     )

    # if all_endpoints:
    #     fig.add_trace(
    #         splt.plot_points(
    #             lifted(all_endpoints, 0.4),
    #             marker={"size": 5, "color": "black", "symbol": "diamond"},
    #         )
    #     )

    # if all_pits:
    #     fig.add_trace(
    #         splt.plot_points(
    #             lifted(all_pits, 1.4),
    #             marker={"size": 7, "color": "red", "symbol": "diamond"},
    #         )
    #     )

    # # Arêtes du maillage (une seule fois)
    # edge_vertices = mesh_vertices[white_mesh.edges_unique]
    # coords = np.full((len(edge_vertices), 3, 3), np.nan)
    # coords[:, :2, :] = edge_vertices

    # fig.add_trace(
    #     go.Scatter3d(
    #         x=coords[:, :, 0].ravel(),
    #         y=coords[:, :, 1].ravel(),
    #         z=coords[:, :, 2].ravel(),
    #         mode="lines",
    #         line={"color": "black", "width": 1},
    #         showlegend=False,
    #     )
    # )

    # fig.write_html(snakemake.output.fig)
