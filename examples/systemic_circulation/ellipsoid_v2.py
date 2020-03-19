# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:09:57 2020

@author: Maaike
"""
import numpy as np
import dolfin as df
from dolfin.cpp.mesh import Point
from mshr import (Box,Ellipsoid)

focus= 4.3  
mesh_resolution=30
inner_eccentricity=0.807075
segments= 30
outer_eccentricity= 0.934819 
cutoff= 2.4

# Compute the inner radii and create the inner ellipsoidal volume.
r1_inner = focus*np.sqrt(1 - inner_eccentricity**2)/inner_eccentricity
r2_inner = focus/inner_eccentricity
inner_volume = Ellipsoid(Point(), r1_inner, r1_inner, r2_inner, segments)

    # Compute the outer radii and create the outer ellipsoidal volume.
r1_outer = focus*np.sqrt(1 - outer_eccentricity**2)/outer_eccentricity
r2_outer = focus/outer_eccentricity
outer_volume = Ellipsoid(Point(), r1_outer, r1_outer, r2_outer, segments)

    # Create a box above the cutoff height and a bit larger than the ellipsoids.
p1 = Point(1.1*r2_outer, 1.1*r2_outer, cutoff)
p2 = Point(-1.1*r2_outer, -1.1*r2_outer, 1.1*r2_outer)
excess = Box(p1, p2)

from matplotlib import pyplot as plt
df.plot(inner_volume)
#plt.show()
plt.savefig('lv_geometry.png')
    # Instantiate and configure the mesh generator.
#    # noinspection PyArgumentList
#generator = CSGCGALMeshGenerator3D()
#generator.parameters['mesh_resolution'] = float(resolution)
#
#    # Enable optimization algorithms.
#generator.parameters['lloyd_optimize'] = True
#generator.parameters['perturb_optimize'] = True