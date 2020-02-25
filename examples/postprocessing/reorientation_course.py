"""
For each cycle, computes the average change in fiber vector
(in degrees) wrt to 1) first cycle and 2) previous cycle.
Saves it to a csv file for further processing.
Runs on a single node.
"""

import os
import glob
import numpy as np
from dolfin import parameters, project, FunctionSpace, Function
from dolfin.cpp.io import XDMFFile

from cvbtk import BiventricleGeometry, read_dict_from_csv, Dataset, save_to_disk
from reorientation import compute_angle_between_vectors, radians_to_degrees


def compute_average_difference(v1, v2):
    """
    Computes average difference in orientation (in radians) between unit vectors v1 and v2.
    """
    v1_array = v1.vector().array().reshape(-1, 3)
    v2_array = v2.vector().array().reshape(-1, 3)

    n = len(v1_array)
    out = 0
    for i in range(n):
        v1_i = v1_array[i, :]
        v2_i = v2_array[i, :]
        cos = np.clip(np.inner(v1_i, v2_i)/(np.linalg.norm(v1_i) * np.linalg.norm(v2_i)), -1, 1)
        acos1 = abs(np.arccos(cos))
        acos2 = abs(np.arccos(-cos))
        out += min(acos1, acos2)

    return out/n


def main():
    parameters['form_compiler']['quadrature_degree'] = 4

    # Specify directory of simulation.
    sim_dir = 'output/original_bayer_method_3'

    # Specify output directory.
    dir_out = sim_dir+'_course'

    # Save diff_prev to a XDMF file for all cycles (to see where in the BiV mesh the change is largest).
    # diff_prev is expressed as a percentage of average diff_prev per cycle.
    save_xdmf = True

    # Find all results.hdf5 files.
    all_dirs = glob.glob(sim_dir + '*')
    all_files = []
    cycles = []
    for d in all_dirs:
        new_files = glob.glob(os.path.join(d, 'results*.hdf5'))
        for f in new_files:
            # Find cycle number.
            c = int(f[:-5].split('_')[-1])
            if c in cycles:
                # Skip.
                continue
            else:
                all_files.append(f)
                cycles.append(c)

    print(all_files)

    # Sort cycles.
    sorted_idx = np.argsort(cycles)

    # Select first file.
    file_init = all_files[sorted_idx[0]]

    # Load geometry.
    inputs = read_dict_from_csv(os.path.join(os.path.split(file_init)[0], 'inputs.csv'))
    inputs['geometry']['load_fiber_field_from_meshfile'] = True
    geometry = BiventricleGeometry(meshfile=file_init, **inputs['geometry'])

    # Create function space (needed if save_xdmf = True
    if save_xdmf:
        Q = FunctionSpace(geometry.mesh(), 'Lagrange', 2)
        normalized_gamma = Function(Q, name='gamma_norm')
        xdmf_file_name = os.path.join(dir_out, 'diff_prev_distribution.xdmf')
        xdmf_file = XDMFFile(xdmf_file_name)

    # Load initial fiber field
    ef_init = geometry.fiber_vectors()[0].to_function(None)

    # Initialize uneven fiber field
    geometry._fiber_vectors = None
    geometry.load_fiber_field(filepath=file_init)
    ef_uneven = geometry.fiber_vectors()[0].to_function(None)

    # ef_even does not yet hold fiber vectors (will be loaded in first iteration).
    ef_even = None

    # Output csv file.
    csv_file_name = os.path.join(dir_out, 'reorientation.csv')

    # Create emtpy dataset.
    out_dataset = Dataset(keys=['cycle', 'diff_init', 'diff_prev'])

    # Add first cycle.
    out_dataset.append(cycle=1, diff_init=0, diff_prev=0)

    # Loop over cycles.
    for i, idx in enumerate(sorted_idx[1:]):

        print('Iteration {} of {}, cycle {}'.format(i+1, len(sorted_idx)-1, cycles[idx]))

        file = all_files[idx]

        # If even iteration, load to ef_even, else load to ef_uneven.
        if np.mod(i, 2) == 0:
            del ef_even
            geometry._fiber_vectors = None
            geometry.load_fiber_field(filepath=file)
            ef_even = geometry.fiber_vectors()[0].to_function(None)

            # Compute average difference fiber vector with initial geometry
            diff_init = compute_average_difference(ef_even, ef_init)

        else:
            del ef_uneven
            geometry._fiber_vectors = None
            geometry.load_fiber_field(filepath=file)
            ef_uneven = geometry.fiber_vectors()[0].to_function(None)

            # Compute average difference fiber vector with initial geometry
            diff_init = compute_average_difference(ef_uneven, ef_init)

        # Compute average difference fiber vector with previous geometry
        diff_prev = compute_average_difference(ef_uneven, ef_even)

        if save_xdmf:  # Prevent division by zero.
            gamma = compute_angle_between_vectors(ef_uneven, ef_even)
            project(radians_to_degrees(gamma), Q, function=normalized_gamma)
            # Save to xmdf
            xdmf_file.write(normalized_gamma, float(cycles[idx]))
            save_to_disk(normalized_gamma, xdmf_file_name[:-5]+'_c{}.xdmf'.format(int(cycles[idx])))

        # Save information to dataset: cycle, diff_init, diff_prev
        out_dataset.append(cycle=cycles[idx], diff_init=diff_init, diff_prev=diff_prev)

    # Save dataset to csv file.
    out_dataset.save(csv_file_name)
    print('Saved to {}'.format(csv_file_name))

    if save_xdmf:
        # Close XDMF file
        xdmf_file.close()
        print('Saved to {}'.format(xdmf_file_name))


if __name__ == '__main__':
    main()
