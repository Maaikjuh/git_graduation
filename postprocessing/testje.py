from dolfin import *
from dolfin.cpp.common import mpi_comm_world
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from celluloid import Camera


result_dir = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/beatit/beatit/results.h5'
mesh_dir = '/mnt/c/Users/Maaike/Documents/Master/Graduation_project/beatit/beatit/utilities/mesh.hdf5'

mesh = Mesh()

openfile = HDF5File(mpi_comm_world(), mesh_dir, 'r')
openfile.read(mesh, 'mesh', False)
V = VectorFunctionSpace(mesh, "CG", 2)

u = Function(V)

focus = 43

theta = 4/10*math.pi
eps_outer = 0.6784
eps_inner = 0.3713

phi_int = 2*math.pi / 8
phi_range = np.arange(-1*math.pi, 1*math.pi, phi_int)

openfile = HDF5File(mpi_comm_world(), result_dir, 'r')

u.set_allow_extrapolation(True)

# # Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure()
ax = plt.axes(xlim=(-40, 40), ylim=(-40, 40))
line, = ax.plot([], [], '-o', lw = 2)

def ellipsoidal_to_cartesian(focus,eps,theta,phi):
    x= focus * math.sinh(eps) * math.sin(theta) * math.cos(phi)
    y = focus * math.sinh(eps) * math.sin(theta) * math.sin(phi)
    z = focus * math.cosh(eps) * math.cos(theta)
    return x, y, z

def init():
    line.set_data([], [])
    return line,

def animate(i):
# for t in range(0, 800):
    vector = 'displacement_{}/vector_0'.format(float(i))
    u = Function(V)
    u.set_allow_extrapolation(True)
    print(vector)
    openfile.read(u, vector)
    posx = []
    posy = []
    for phi in phi_range: 

        x_epi, y_epi, z_epi = ellipsoidal_to_cartesian(focus, eps_outer, theta, phi)
        tau = z_epi/(focus * math.cosh(eps_inner))
        theta_inner = math.acos(tau)
        x_endo, y_endo, z_endo = ellipsoidal_to_cartesian(focus,eps_inner,theta_inner,phi)

        x_mid = (x_epi + x_endo)/2
        y_mid = (y_epi + y_endo)/2
        z_mid = (z_epi + z_endo)/2

        for ii, points in enumerate([[x_epi, y_epi, z_epi], [x_mid, y_mid, z_mid], [x_endo, y_endo, z_endo]]):
            # print(ii)
            xu, yu, zu = u(points)
            x = points[0] + xu
            y = points[1] + yu
            z = points[2] + zu

            posx.append(x)
            posy.append(y)

            # ax.clear()
            # ax.plot(x,y)
            line.set_data(x,y)

    # return line,

time = range(0, 800)
anim = animation.FuncAnimation(fig, animate, frames = 800)
print('saving')
plt.show()
anim.save('rotation.mp4')