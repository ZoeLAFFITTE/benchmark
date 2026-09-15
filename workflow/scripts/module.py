def write_vtk_fs_ascii(filename, coords, faces):
    """
    Écrit un fichier VTK 1.0 ASCII compatible FreeSurfer.
    coords : (N,3) numpy array
    faces  : (M,3) numpy array
    """
    n_points = coords.shape[0]
    n_faces = faces.shape[0]

    with open(filename, 'w') as f:
        # Header VTK 1.0 ASCII
        f.write("# vtk DataFile Version 1.0\n")
        f.write("vtk output\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n_points} float\n")

        # Points : un par ligne
        for p in coords:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

        # Polygons : triangles
        total_ints = n_faces * 4  # 3 indices + 1 nombre par triangle
        f.write(f"POLYGONS {n_faces} {total_ints}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

    print(f"VTK FreeSurfer ASCII écrit : {filename}")