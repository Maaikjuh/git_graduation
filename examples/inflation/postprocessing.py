"""
This script performs some minor postprocessing of the inflation simulation.

Not tested if this runs in parallel.
"""

from dolfin import VectorFunctionSpace, Function
from dolfin.cpp.common import mpi_comm_self
from dolfin.cpp.io import HDF5File, XDMFFile

from cvbtk import LeftVentricleGeometry, LeftVentricleModel, save_to_disk, Dataset, print_once

# Specify the paths to the HDF5 (displacement) and csv (hemodynamics) results files.
results_hdf5_filename = 'results.hdf5'
results_csv_filename = 'results.csv'

# Load geometry.
geometry = LeftVentricleGeometry(meshfile=results_hdf5_filename,
                                 load_fiber_field_from_meshfile=False)

# Create a DOLFIN FunctionSpace for the unknowns:
V = VectorFunctionSpace(geometry.mesh(),
                        'Lagrange', 2)

# Create the DOLFIN Function objects for the unknowns.
u = Function(V, name='displacement')

# save tags.
save_to_disk(geometry.tags(), 'tags.xdmf')

# Load csv dataframe.
data = Dataset(filename=results_csv_filename)
if 'vector_number' in data.keys():
    vector_numbers = list(data['vector_number'])
else:
    vector_numbers = list(range(len(data['plv'])))

# Open XDMF file to save to.
with XDMFFile('u.xdmf') as f_xdmf:
    # Loop over vectors.
    with HDF5File(mpi_comm_self(), results_hdf5_filename, 'r') as f_hdf5:
        for idx, vector in enumerate(vector_numbers):
            print('{0:.2f} %'.format(idx / len(vector_numbers) * 100))

            # Read displacement of timestep to function u.
            u_vector = 'displacement/vector_{}'.format(vector)
            f_hdf5.read(u, u_vector)

            # Save function u to the XDMF file.
            f_xdmf.write(u, float(idx))  # Timstamp needs to be a float!

print('Done!')
