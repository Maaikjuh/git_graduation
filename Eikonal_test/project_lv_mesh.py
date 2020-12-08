from dolfin import *
from dolfin.cpp.common import mpi_comm_world
import numpy as np
import os
# from fenicstools import interpolate_nonmatching_mesh

# #eikonal mesh
mesh = Mesh()


meshres = 80 

# filename = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/meshes/Eikonal_meshes/seg_20_mesh_20_bue/td.hdf5'
# filename = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/meshes/Eikonal_meshes/15-06_10-06_mesh_50_purk_fac_kot00/td.hdf5'
# eikonal_dir = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/Eikonal_meshes/seg_20_mesh_80_bue_purk_kot_exp_epi_border_div3_8/td.hdf5'.format(meshres)
# mesh_dir = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/lv_maaike_seg20_res{}_fibers_mesh.hdf5'.format(meshres)
# ls_dir = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/15-09_new_vars_ref_cyc_5_res_30/cycle_5_begin_ic_ref/ls.hdf5'
# lc_dir = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/Results/leftventricular model/15-09_new_vars_ref_cyc_5_res_30/cycle_5_begin_ic_ref/lc.hdf5'

file_name = 'kerckhoffs_ischemic_div5_LAD_k_1'
file_name = 'kerckhoffs_k_1'
eikonal_dir = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/Eikonal_meshes/seg_20_mesh_50_{}/td.hdf5'.format(file_name)
# mesh_dir = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/lv_maaike_seg20_res30_fibers_mesh.hdf5'
# dir_out = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/Eikonal_meshes/seg_20_mesh_30_{}'.format(file_name)
# if MPI.rank(mpi_comm_world()) == 0:

# openfile = HDF5File(mpi_comm_world(), eikonal_dir, 'r')
# openfile1 = HDF5File(mpi_comm_world(), mesh_dir, 'r')
# openfile1.read(mesh, 'mesh', False)
# V_project = FunctionSpace(mesh, "Lagrange", 2)

mesh = Mesh()
openfile = HDF5File(mpi_comm_world(), eikonal_dir, 'r')
openfile.read(mesh, 'mesh', False)
V_eikonal = FunctionSpace(mesh, "Lagrange", 2)
print('num cells: ', mesh.num_cells())
print('max size: ', mesh.hmax())
print('min size: ', mesh.hmin())


parameters['allow_extrapolation'] = True
td = Function(V_eikonal, name = 'td')

openfile.read(td, 'td/vector_0')
td.set_allow_extrapolation(True)

focus = 4.3
rastr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+{f})*(x[2]+{f}))".format(f=focus)
rbstr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-{f})*(x[2]-{f}))".format(f=focus)

sigmastr="(1./(2.*{f})*({ra}+{rb}))".format(ra=rastr,rb=rbstr,f=focus)
eps = Expression("acosh({sigma})".format(sigma=sigmastr),degree=3,element=V_eikonal.ufl_element())

td_inner_cpp = "(eps <= {eps_endo})? td : 0".format(eps_endo = 0.3713)
td_inner_exp = Expression(td_inner_cpp,degree=3,element=V_eikonal.ufl_element(), eps=eps, td=td)
td_endo = Function(V_eikonal)
td_endo.interpolate(td_inner_exp)
print('Endocard activated in {max_endo}ms, whole LV activated in {max_lv}ms'.format(
    max_endo = round(max(td_endo.vector()), 2), max_lv = round(max(td.vector()), 2)))
    

# td_project = project(td, V_project)
# td_project.rename('td','td')

# # write values to xdmf for easy viewing. 
# ofile = XDMFFile(mpi_comm_world(), os.path.join(dir_out,"td_solution.xdmf"))
# ofile.write(td_project)

# # save mesh and solution for td as hdf5 to be used later on in the model
# with HDF5File(mpi_comm_world(), os.path.join(dir_out,'td.hdf5'), 'w') as f:
#     f.write(td_project, 'td')
#     f.write(mesh, 'mesh')

# focus = 4.3
# # get eps and phi values of the nodes in the mesh
# rastr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+{f})*(x[2]+{f}))".format(f=focus)
# rbstr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-{f})*(x[2]-{f}))".format(f=focus)

# sigmastr="(1./(2.*{f})*({ra}+{rb}))".format(ra=rastr,rb=rbstr,f=focus)

# eps = Expression("acosh({sigma})".format(sigma=sigmastr),degree=3,element=V.ufl_element())
# phi = Expression("atan2(x[1],x[0])",degree=3,element=V.ufl_element())

# td_inner = "(eps <= 0.3713)? td : 0"
# td_exp = Expression(td_inner,degree=3,element=V.ufl_element(), eps=eps, td=td)
# td_endo = Function(V)
# td_endo.interpolate(td_exp)
# print(max(td_endo.vector()))
# openfile = HDF5File(mpi_comm_world(), ls_dir, 'r')
# parameters['allow_extrapolation'] = True
# ls = Function(V)
# openfile.read(ls, 'ls/vector_0')

# openfile = HDF5File(mpi_comm_world(), lc_dir, 'r')
# parameters['allow_extrapolation'] = True
# lc = Function(V)
# openfile.read(lc, 'lc/vector_0')

# T0 = Function(V, name='T0')
# T0.assign(Constant(160.0))

# file = XDMFFile('td_pre_project.xdmf')
# file.write(td)

# MPI.barrier(mpi_comm_world())
# mesh = Mesh(comm)
# openfile = HDF5File(mpi_comm_world(), mesh_dir, 'r')
# openfile.read(mesh, 'mesh', False)
# V = FunctionSpace(mesh, "Lagrange", 2)
# V = Function(V)
# parameters['allow_extrapolation'] = False

# build_module("tact = project(-1*td,V)")
# if MPI.rank(mpi_comm_world()) == 1:
# MPI.barrier(mpi_comm_world())
# tact = project(-1*td,V)
# # MPI.barrier(mpi_comm_world())
# # print('saving')
# tact.rename('eikonal', 'eikonal')
# file = XDMFFile('td_post_project.xdmf')
# file.write(tact)

# # td.set_allow_extrapolation(True)
# V.interpolate(td)
# u2 = interpolate_nonmatching_mesh(td, V)
# file = File('eikonal.pvd')
# file << tact
# tact_dummy = tact
# file = XDMFFile('eikonal_td_new.xdmf')
# for t in range(1,50,2):
#     tact.rename('eikonal', 'eikonal')
#     print(t)
#     if t == 25:
#         # tact_dummy.vector()[:] += float(-20)
#         tact.vector()[:] += float(-20)
#     else:
#         # tact_dummy.vector()[:] += float(2)
#         tact.vector()[:] += float(2)
#     # tact.assign(tact_dummy)
    
#     file.write(tact, float(t))
# file.close()
# mesh = V.ufl_domain().ufl_cargo()

# # Extract the element parameters from the given vector function space.
# family = V.ufl_element().family()
# degree = V.ufl_element().degree()
# print(mesh, family, degree)
# lc = 1.
# ls = 2.
# iso_cond = conditional(ge(lc, 1.5), 1, 0)
# #f_iso = iso_cond*iso_term
# T0 = 160.
# # 17-03, substitute T0 with self.T0
# iso_term = T0*(tanh(2.0*(lc - 1.5)))**2
# f_iso = iso_cond*iso_term

# tp = project(400., V)
# t_max =  160.0*(ls +0.5)
# print(t_max)
# tact.vector()[:] = t_max
# print(min(tp.vector()), max(tp.vector()))
# twitch_term_1 = (tanh(tact/75.0))**2
# twitch_term_2 = (tanh((t_max - tact)/150.0))**2

# twitch_cond_1 = ge(tact, 0)
# twitch_cond_2 = le(tact, t_max)
# twitch_cond = conditional(And(twitch_cond_1, twitch_cond_2), 1, 0)
# f_twitch = twitch_cond*twitch_term_1*twitch_term_2

# projected = project(f_twitch, V)
# p = f_iso*f_twitch*20.0*(ls - lc)