# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:35:31 2020

@author: Maaike
"""
#import numpy as np
#from pyellipsoid import drawing
#import matplotlib.pyplot as plt
#
## Define an image shape, axis order is: Z, Y, X
#image_shape = (128, 128, 128)
#
## Define an ellipsoid, axis order is: X, Y, Z
#ell_center = (64, 64, 64)
#ell_radii = (5, 50, 30)
#ell_angles = np.deg2rad([80, 30, 20]) # Order of rotations is X, Y, Z
#
## Draw a 3D binary image containing the ellipsoid
#print(ell_angles)
#image = drawing.make_ellipsoid_image(image_shape, ell_center, ell_radii, ell_angles)
#imgplot = plt.imshow(image)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig.add_subplot(111, projection='3d')

coefs = (0.39,0.39,1)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
# Radii corresponding to the coefficients:
rx, ry, rz = 1/np.sqrt(coefs)

C = 1
eps = 0.37
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)

phi = np.linspace(0,2*np.pi, 256).reshape(256, 1) # the angle of the projection in the xy-plane
theta = np.linspace(0, np.pi, 256).reshape(-1, 256) # the angle from the polar axis, ie the polar angle



# Set of all spherical angles:
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# Cartesian coordinates that correspond to the spherical angles:
# (this is the equation of an ellipsoid):
#x = C * np.sinh(eps)*np.sin(theta)*np.cos(phi)
#y = C * np.sinh(eps)*np.sin(theta)*np.sin(phi)
#z = C * np.cosh(eps), np.cos(theta)

x = rx * np.outer(np.cos(u), np.sin(v))
y = ry * np.outer(np.sin(u), np.sin(v))
z = rz * np.outer(np.ones_like(u), np.cos(v))

x = 0.39*np.sin(theta)*np.cos(phi)
y = 0.39*np.sin(theta)*np.sin(phi)
z = 0.43*np.cos(theta)

fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b')
ax.set(xlim=(-1, 1), ylim=(-1, 1)) 
# Plot:
#ax.plot_surface(x, y, z,  rstride=4, cstride=4)
# Adjustment of the axes, so that they all have the same span:
#max_radius = max(rx, ry, rz)
#for axis in 'xyz':
#    getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

#plt.show()
