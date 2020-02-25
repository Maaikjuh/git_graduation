"""
This script demonstrates how the cardiac vectors are implemented.

NOTE: Postprocessing (plots and printed statsistics) only works when running on 1 processor
(due to partitioning of the mesh when using parallelization with MPI).
So run this script on one processor.
"""

from dolfin import Function, project, VectorFunctionSpace, parameters, sqrt, FunctionSpace
import matplotlib.pyplot as plt
from dolfin.cpp.common import MPI, mpi_comm_world
from dolfin.cpp.function import FunctionAssigner
from dolfin.cpp.io import XDMFFile, HDF5File
import os
import cvbtk.resources
import numpy as np
from cvbtk import BasisVectorsBayer, CoordinateSystem, \
    vector_space_to_scalar_space, BiventricleGeometry, BiventricleModel, save_to_disk, reset_values, figure_make_up, \
    quadrature_function_space
import time

plt.close('all')


def analyze_basis_vectors(geometry, V, dir_out='.', fontsize=13, name=''):

    # Create output directory.
    if MPI.rank(mpi_comm_world()) == 0:
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
    # Synchronize.
    MPI.barrier(mpi_comm_world())

    # Create scalar function space.
    Q = vector_space_to_scalar_space(V)

    print('Creating cardiac and fiber vectors... ')
    ec_f, el_f, et_f = [e.to_function(V) for e in geometry.cardiac_vectors()]
    ef_f, es_f, en_f = [e.to_function(V) for e in geometry.fiber_vectors()]
    ah_f, at_f = [a.to_function(Q) for a in geometry.fiber_angles()]
    u_f, v_epi_f, v_endo_f = [_.to_function(FunctionSpace(geometry.mesh(), 'Lagrange', 2))
                              for _ in geometry.wallbounded_coordinates()]

    # Project the norm of the fiber vectors. This is a good test to check
    # if the fiber vectors have unit length at the quadrature points
    # (then the projected norm should equal one across the entire mesh).
    save_to_disk(project(sqrt(ef_f[0]*ef_f[0] + ef_f[1]*ef_f[1] + ef_f[2]*ef_f[2]),
                         FunctionSpace(geometry.mesh(), 'Lagrange', 2)),
                 os.path.join(dir_out, 'norm_ef.xdmf'))

    # Check if functions are quadrature functions and project them on a Lagrange space if so in order to be able to
    # save and them for visualization and to compute values at arbitrary locations in the mesh
    if V.ufl_element().family() == 'Quadrature':

        # Create 2nd degree Lagrange function spaces.
        V_lg = VectorFunctionSpace(geometry.mesh(), 'Lagrange', 2)
        Q_lg = vector_space_to_scalar_space(V_lg)

        # Project vector Functions.
        ec_f, el_f, et_f, ef_f, es_f, en_f = [
            project(qf, V_lg) for qf in [ec_f, el_f, et_f, ef_f, es_f, en_f]
        ]

        # Project scalar Functions.
        ah_f, at_f, u_f, v_epi_f, v_endo_f = [
            project(qf, Q_lg) for qf in [ah_f, at_f, u_f, v_epi_f, v_endo_f]
            ]

    # Get analytical functions of ah and at. They take as input (u, v).
    ah_anal, at_anal = geometry._fiber_angle_functions()

    # The vectors can be exported to XDMF files for visualization in ParaView.
    save_to_disk(ec_f, os.path.join(dir_out, 'ec.xdmf'))
    save_to_disk(el_f, os.path.join(dir_out, 'el.xdmf'))
    save_to_disk(et_f, os.path.join(dir_out, 'et.xdmf'))

    save_to_disk(ef_f, os.path.join(dir_out, 'ef.xdmf'))
    save_to_disk(es_f, os.path.join(dir_out, 'es.xdmf'))
    save_to_disk(en_f, os.path.join(dir_out, 'en.xdmf'))

    save_to_disk(ah_f, os.path.join(dir_out, 'ah.xdmf'))
    save_to_disk(at_f, os.path.join(dir_out, 'at.xdmf'))

    # Compare normalized local coordinates with actual travelled distance.
    # Plot normalized transmural coordinate against normalized travelled distance at
    # lv at y=z=0
    r_inner = geometry.parameters['geometry']['R_1']
    r_outer = geometry.parameters['geometry']['R_2']
    n = 200
    x_coords = np.linspace(r_inner - 1e-3, r_outer + 1e-3, n)
    d_transmural = []  # I used both d and v for the transmural coordinate...
    true_distance = []
    effective_ah = []
    effective_at = []
    u_list = []
    for x in x_coords:
        try:
            # Transmural coordinate as defined on mesh.
            d_transmural.append(v_epi_f(x, 0, 0))

            # True transmural distance.
            true_distance.append(x - x_coords[0])

            # Compute/approximate effectiv helix and transmural angles.
            effective_ah.append(ah_f(x, 0, 0))
            effective_at.append(at_f(x, 0, 0))

            # Longitudnal coordinate as defined on mesh.
            u_list.append(u_f(x, 0, 0))
        except:
            # Catch error if point falls outside mesh.
            pass
    # Compute true normalized distance/coordinate.
    true_distance-=true_distance[0]
    true_distance = abs(true_distance)
    true_distance/=max(true_distance)
    true_distance = 2*true_distance - 1

    # Compute the helix angles given the transmural and longitudinal normalized coordinates.
    true_ah = []
    true_at = []
    for u, v in zip(u_list, true_distance):
        true_ah.append(ah_anal(u, v)*180/np.pi)
        true_at.append(at_anal(u, v)*180/np.pi)

    # Plots.
    # Compare true transmural distance to transmural distance defined on mesh.
    plt.figure('v')
    plt.plot(true_distance, d_transmural, label='LV free wall')

    # Compare effective helix angle with true/expected helix angle.
    plt.figure('ah_lv')
    plt.plot(true_distance, true_ah, label=r'Analytical $\alpha_h$')
    plt.plot(true_distance, effective_ah, '--', label=r'Effective $\alpha_h$')
    figure_make_up(title='LV', xlabel='Analytical transmural coordinate [-]',
                   ylabel='Helix angle [$^o$]', fontsize=fontsize)
    plt.savefig(os.path.join(dir_out, 'ah_anal_vs_effective_lv.png'), dpi=300)

    # Compare effective transmural angle with true/expected transmural angle.
    plt.figure('at_lv')
    plt.plot(true_distance, true_at, label=r'Analytical $\alpha_t$')
    plt.plot(true_distance, effective_at, '--', label=r'Effective $\alpha_t$')
    figure_make_up(title='LV' , xlabel='Analytical transmural coordinate [-]',
                   ylabel='Transverse angle [$^o$]', fontsize=fontsize)
    plt.savefig(os.path.join(dir_out, 'at_anal_vs_effective_lv.png'), dpi=300)

    # Make true distance vs local coordinate u figure current for next plot commands.
    plt.figure('v')

    # Now we will repeat the above procedure for different locations in the mesh.

    # Plot normalized transmural coordinate d against travelled distance at
    # lv at x=z=0
    r_inner = geometry.parameters['geometry']['R_1']
    r_outer = geometry.parameters['geometry']['R_2']
    y_coords = np.linspace(r_inner - 1e-3, r_outer + 1e-3, n)
    d_transmural = []
    true_distance = []
    for y in y_coords:
        try:
            d_transmural.append(v_epi_f(0,y,0))
            true_distance.append(y - y_coords[0])
        except:
            pass
    true_distance-=true_distance[0]
    true_distance = abs(true_distance)
    true_distance/=max(true_distance)
    true_distance = 2*true_distance - 1
    plt.plot(true_distance, d_transmural, '--m', label='LV free wall')

    # Plot normalized transmural coordinate d against travelled distance at
    # lv apex at x=y=0
    r_inner = -geometry.parameters['geometry']['Z_1']
    r_outer = -geometry.parameters['geometry']['Z_2']
    z_coords = np.linspace(r_inner + 1e-3, r_outer - 1e-3, n)
    d_transmural = []
    true_distance = []
    for z in z_coords:
        try:
            d_transmural.append(v_epi_f(0,0,z))
            true_distance.append(abs(z - z_coords[0]))
        except:
            pass
    true_distance -= true_distance[0]
    true_distance = abs(true_distance)
    true_distance/=max(true_distance)
    true_distance = 2*true_distance - 1
    plt.plot(true_distance, d_transmural, '--r', label='LV apex')

    # Plot normalized transmural coordinate d against travelled distance at
    # rv at y=z=0
    r_inner = -geometry.parameters['geometry']['R_3']
    r_outer = -geometry.parameters['geometry']['R_4']
    x_coords = np.linspace(r_inner + abs(r_inner/10), r_outer - 1e-3, n)
    d_transmural = []
    true_distance = []
    effective_ah = []
    effective_at = []
    u_list = []
    for x in x_coords:
        try:
            d_transmural.append(v_epi_f(x, 0, 0))
            true_distance.append(abs(x - x_coords[0]))

            effective_ah.append(ah_f(x, 0, 0))
            effective_at.append(at_f(x, 0, 0))
            u_list.append(u_f(x, 0, 0))
        except:
            pass
    true_distance-=true_distance[0]
    true_distance = abs(true_distance)
    true_distance/=max(true_distance)
    true_distance = 2*true_distance - 1

    true_ah = []
    true_at = []
    for u, v in zip(u_list, true_distance):
        true_ah.append(ah_anal(u, v)*180/np.pi)
        true_at.append(at_anal(u, v)*180/np.pi)

    plt.figure('v')
    plt.plot(true_distance, d_transmural, '--g', label='RV free wall')

    plt.figure('ah_rv')
    plt.plot(true_distance, true_ah, label=r'Analytical $\alpha_h$')
    plt.plot(true_distance, effective_ah, '--', label=r'Effective $\alpha_h$')
    figure_make_up(title='RV', xlabel='Analytical transmural coordinate [-]',
                   ylabel='Helix angle [$^o$]', fontsize=fontsize)
    plt.savefig(os.path.join(dir_out, 'ah_anal_vs_effective_rv.png'), dpi=300)

    plt.figure('at_rv')
    plt.plot(true_distance, true_at, label=r'Analytical $\alpha_t$')
    plt.plot(true_distance, effective_at, '--', label=r'Effective $\alpha_t$')
    figure_make_up(title='RV' , xlabel='Analytical transmural coordinate [-]',
                   ylabel='Transverse angle [$^o$]', fontsize=fontsize)
    plt.savefig(os.path.join(dir_out, 'at_anal_vs_effective_rv.png'), dpi=300)

    # Make true distance vs local coordinate u figure current.
    plt.figure('v')

    # Plot normalized transmural coordinate d against travelled distance at
    # septum at y=z=0
    r_inner = -geometry.parameters['geometry']['R_1sep']
    r_outer = -geometry.parameters['geometry']['R_2sep']
    x_coords = np.linspace(r_inner + abs(r_inner/10), r_outer - 1e-3, n)
    d_transmural = []
    true_distance = []
    effective_ah = []
    effective_at = []
    u_list = []
    for x in x_coords:
        try:
            d_transmural.append(v_endo_f(x, 0, 0))
            true_distance.append(abs(x - x_coords[0]))

            effective_ah.append(ah_f(x, 0, 0))
            effective_at.append(at_f(x, 0, 0))
            u_list.append(u_f(x, 0, 0))
        except:
            pass
    true_distance-=true_distance[0]
    true_distance = abs(true_distance)
    true_distance/=max(true_distance)
    true_distance = 2*true_distance - 1

    true_ah = []
    true_at = []
    for u, v in zip(u_list, true_distance):
        true_ah.append(ah_anal(u, v)*180/np.pi)
        true_at.append(at_anal(u, v)*180/np.pi)

    plt.figure('v')
    plt.plot(true_distance, d_transmural, '--c', label='Septum')

    plt.figure('ah_sep')
    plt.plot(true_distance, true_ah, label=r'Analytical $\alpha_h$')
    plt.plot(true_distance, effective_ah, '--', label=r'Effective $\alpha_h$')
    figure_make_up(title='Septum', xlabel='Analytical transmural coordinate [-]',
                   ylabel='Helix angle [$^o$]', fontsize=fontsize)
    plt.savefig(os.path.join(dir_out, 'ah_anal_vs_effective_sep.png'), dpi=300)

    plt.figure('at_sep')
    plt.plot(true_distance, true_at, label=r'Analytical $\alpha_t$')
    plt.plot(true_distance, effective_at, '--', label=r'Effective $\alpha_t$')
    figure_make_up(title='Septum' , xlabel='Analytical transmural coordinate [-]',
                   ylabel='Transverse angle [$^o$]', fontsize=fontsize)
    plt.savefig(os.path.join(dir_out, 'at_anal_vs_effective_sep.png'), dpi=300)

    # Make true distance vs local coordinate u figure current.
    plt.figure('v')

    plt.plot([-1,1], [-1,1], ':k', label='x=y')
    plt.legend(fontsize=fontsize-2)
    plt.tick_params(labelsize=fontsize - 2)
    plt.xlabel('Analytical transmural coordinate [-]', fontsize=fontsize)
    plt.ylabel('Derived transmural coordinate [-]', fontsize=fontsize)
    plt.savefig(os.path.join(dir_out, 'Transmural_distance.png'), dpi=300)


# Set the quadrature degree.
parameters['form_compiler']['quadrature_degree'] = 4

# Set the degree of the Lagrange function spaces / elements.
degree = 2

# Fontsize for plotting.
fontsize= 14

# Compute and analyse cardiac and fiber vectors of the adapted Bayer method.
print('Adapted method.')

# Specify the output directory.
dir_out = r'output/cardiac_vectors_biv/adapted'

# Inputs for adapted method.
bayer_inputs_3 = {
    'apex_bc_mode': 'line',
    'interp_mode': 3,
    'retain_mode': 'combi',
    'correct_ec': True,
    'linearize_u': True,
    'mirror': True,
    'transmural_coordinate_mode': 2,
    'verbose': True,
    'dir_out': dir_out}

# We shouldn;t load any fiber field, as we investigate how the fiber field differs
# between different methods with a rule-based fiber field.
inputs_3 = {
    'load_fiber_field_from_meshfile': False,
    'bayer': bayer_inputs_3}

# Load biventricle geometry.
geometry_3 = cvbtk.resources.reference_biventricle(**inputs_3)

# Create vector function space to define the vectors on.
V_3 = VectorFunctionSpace(geometry_3.mesh(), 'Lagrange', degree)

# Analyze vectors (creates some figures and XDMF files.
analyze_basis_vectors(geometry_3, V_3, dir_out=dir_out, fontsize=fontsize, name='')

plt.close('all')

# Repeat the analysis, but used the original Bayer method now.
print('Original method.')

# Specify output directory.
dir_out = r'output/cardiac_vectors_biv/original_quad'

# Inputs for original Bayer method.
bayer_inputs_1 = {
    'apex_bc_mode': 'point',
    'interp_mode': 1,
    'retain_mode': 'el',
    'correct_ec': False,
    'linearize_u': True,
    'mirror': False,
    'transmural_coordinate_mode': 1,
    'verbose': True,
    'dir_out': dir_out}

inputs_1 = {
    'load_fiber_field_from_meshfile': False,
    'bayer': bayer_inputs_1}

# Load biventricle geometry.
geometry_1 = cvbtk.resources.reference_biventricle(**inputs_1)

# Create vector function space.
V_1 = VectorFunctionSpace(geometry_1.mesh(), 'Lagrange', degree)

analyze_basis_vectors(geometry_1, V_1, dir_out=dir_out, fontsize=fontsize, name='Original')
plt.close('all')

# Repeat again for the adapted method, but now define the vectors on quadrature elements.
print('Adapted method quad.')

dir_out = r'output/cardiac_vectors_biv/adapted_quad'

bayer_inputs_2 = {
    'apex_bc_mode': 'line',
    'interp_mode': 3,
    'retain_mode': 'combi',
    'correct_ec': True,
    'linearize_u': True,
    'mirror': True,
    'transmural_coordinate_mode': 2,
    'verbose': True,
    'dir_out': dir_out}

inputs_2 = {
    'load_fiber_field_from_meshfile': False,
    'bayer': bayer_inputs_2}

# Load geometry.
geometry_2 = cvbtk.resources.reference_biventricle(**inputs_2)

# Create quadrature vector function space.
V_2 = quadrature_function_space(geometry_2.mesh())

# Analyze, notice that the projected norm of the fiber vector (see norm_ef.xdmf)
# is now equal to 1 across the entire mesh.
analyze_basis_vectors(geometry_2, V_2, dir_out=dir_out, fontsize=fontsize, name='Adapted')
plt.close('all')
