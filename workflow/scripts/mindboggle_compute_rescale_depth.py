import sys
from mindboggle.guts.mesh import rescale_by_neighborhood

input_vtk = sys.argv[1]
output_vtk = sys.argv[2]

rescaled, output_file = rescale_by_neighborhood(
    input_vtk=input_vtk,
    indices=[],
    nedges=10,
    p=99,
    set_max_to_1=True,
    save_file=True,
    output_filestring=output_vtk,
    background_value=-1
)