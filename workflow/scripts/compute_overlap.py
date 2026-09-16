sulcus_to_basin = {}

filtered_sulci_texture = np.zeros_like(sulci_texture)

for sulcus_label in np.unique(sulci_texture):

    # Ignorer le fond
    if sulcus_label == 0:
        continue

    # Sommets de ce sillon
    sulcus_mask = sulci_texture == sulcus_label

    # Bassins présents sous ce sillon
    basin_labels = basins_texture[sulcus_mask]

    # Trouver le bassin qui couvre le plus de sommets du sillon
    labels, counts = np.unique(
        basin_labels,
        return_counts=True,
    )

    covering_basin = labels[np.argmax(counts)]

    sulcus_to_basin[sulcus_label] = covering_basin

    # Garder le sillon si son bassin principal est sélectionné
    if covering_basin in filtered_basins:
        filtered_sulci_texture[sulcus_mask] = sulcus_label