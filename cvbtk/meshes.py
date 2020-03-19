# -*- coding: utf-8 -*-
"""
This module provides functions that mimic the built-in DOLFIN mesh classes.
"""
import numpy as np
from scipy import interpolate
from dolfin.cpp.mesh import Point
from dolfin.cpp.common import Parameters
from mshr import (Box, CSGCGALDomain3D, CSGCGALMeshGenerator3D, Cylinder,
                  Ellipsoid, Sphere)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = [
    'LeftVentricleMesh',
    'LeftVentricleVADMesh',
    'BiventricleMesh',
    'compute_biventricular_parameters',
    'ThickWalledSphereMesh'
]


def LeftVentricleMesh(focus, cutoff, inner_eccentricity, outer_eccentricity,
                      resolution, **kwargs):
    """
    Tetrahedral mesh of an idealized left ventricle defined as a pair of
    truncated, confocal, prolate spheroids/ellipsoids.

    Args:
        focus: Distance from the origin to the shared focus points.
        cutoff: Distance above the xy-plane to truncate the ellipsoids.
        inner_eccentricity: Eccentricity of the inner ellipsoid.
        outer_eccentricity: Eccentricity of the outer ellipsoid.
        resolution: Mesh resolution parameter.
        **kwargs (optional): Arbitrary keyword arguments for the mesh generator.

    Returns:
        :class:`~dolfin.Mesh`
    """
    # Check if segments were given.
    segments = int(kwargs.get('segments', 20))

    # Compute the inner radii and create the inner ellipsoidal volume.
    r1_inner = focus*np.sqrt(1 - inner_eccentricity**2)/inner_eccentricity
    r2_inner = focus/inner_eccentricity
    inner_volume = Ellipsoid(Point(), r1_inner, r1_inner, r2_inner, segments=segments)

    # Compute the outer radii and create the outer ellipsoidal volume.
    r1_outer = focus*np.sqrt(1 - outer_eccentricity**2)/outer_eccentricity
    r2_outer = focus/outer_eccentricity
    outer_volume = Ellipsoid(Point(), r1_outer, r1_outer, r2_outer, segments=segments)

    # Create a box above the cutoff height and a bit larger than the ellipsoids.
    p1 = Point(1.1*r2_outer, 1.1*r2_outer, cutoff)
    p2 = Point(-1.1*r2_outer, -1.1*r2_outer, 1.1*r2_outer)
    excess = Box(p1, p2)

    # Instantiate and configure the mesh generator.
    # noinspection PyArgumentList
    generator = CSGCGALMeshGenerator3D()
    generator.parameters['mesh_resolution'] = float(resolution)

    # Enable optimization algorithms.
    generator.parameters['lloyd_optimize'] = True
    generator.parameters['perturb_optimize'] = True

    # Combine the domains into a single geometry, mesh, and return.
    domain = CSGCGALDomain3D(outer_volume - inner_volume - excess)
    return generator.generate(domain)


def LeftVentricleVADMesh(focus, cutoff, inner_eccentricity, outer_eccentricity,
                         lvad_tube_radius, resolution, **kwargs):
    """
    Tetrahedral mesh of an idealized left ventricle defined as a pair of
    truncated, confocal, prolate spheroids/ellipsoids.

    Args:
        focus: Distance from the origin to the shared focus points.
        cutoff: Distance above the xy-plane to truncate the ellipsoids.
        inner_eccentricity: Eccentricity of the inner ellipsoid.
        outer_eccentricity: Eccentricity of the outer ellipsoid.
        lvad_tube_radius: Radius of the LVAD insert tube.
        resolution: Mesh resolution parameter.
        **kwargs (optional): Arbitrary keyword arguments for the mesh generator.

    Returns:
        :class:`~dolfin.Mesh`
    """
    # Check if segments were given.
    segments = int(kwargs.get('segments', 30))

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

    # Create the LVAD tube insert.
    tube = Cylinder(Point(0, 0, 1.1*r2_outer), Point(0, 0, -1.1*r2_outer),
                    lvad_tube_radius, lvad_tube_radius, segments=64)

    # Instantiate and configure the mesh generator.
    # noinspection PyArgumentList
    generator = CSGCGALMeshGenerator3D()
    generator.parameters['mesh_resolution'] = float(resolution)
    generator.parameters['feature_threshold'] = 45.0  # default: 70.0
    # TODO Implement way to check/configure additional parameters.

    # Combine the domains into a single geometry, mesh, and return.
    domain = CSGCGALDomain3D((outer_volume - inner_volume) - (excess + tube))
    return generator.generate(domain)


def BiventricleMesh(geometry_parameters, **kwargs):
    """
    Creates a biventricular mesh.
    Args:
        geometry_parameters: Dolfin parameter set with geometric parameters
                            (radii of ellipsoids and truncation height).
        **kwargs: Arbitrary keyword arguments for segments and meshing parameters.
    Returns:
        dolfin Mesh object.
    """
    # ------------------------------------------------------------------- #
    # Extract geometric parameters
    # ------------------------------------------------------------------- #
    R_1 = geometry_parameters['R_1']
    Z_1 = geometry_parameters['Z_1']
    R_2 = geometry_parameters['R_2']
    Z_2 = geometry_parameters['Z_2']
    R_1sep = geometry_parameters['R_1sep']
    R_2sep = geometry_parameters['R_2sep']
    R_3 = geometry_parameters['R_3']
    R_3y = geometry_parameters['R_3y']
    Z_3 = geometry_parameters['Z_3']
    R_4 = geometry_parameters['R_4']
    Z_4 = geometry_parameters['Z_4']
    h = geometry_parameters['h']

    center_rv_x = -0.000001*R_1  # if center_rv_x = 0.0 then CSGCGALMeshGenerator3D() crashes for unknown reason.
    # ------------------------------------------------------------------- #
    # Create domain                                                       #
    # ------------------------------------------------------------------- #
    print('Creating mesh...')
    # Extract segments parameter (if given).
    segments = int(kwargs.get('segments', 20))

    # LV endocardium
    cyl_lv_endo_fw = Ellipsoid(Point(0.0, 0.0, 0.0), R_1, R_1, Z_1, segments=segments)  # free wall
    box_lv_fw = Box(Point(-1.1 * R_2, -1.1 * R_2, -1.1 * Z_2),
                    Point(0., 1.1 * R_2, 1.1 * Z_2))  # box to remove left half of the LV free wall
    cyl_lv_endo_sw = Ellipsoid(Point(0.0, 0.0, 0.0), R_1sep, R_1, Z_1, segments=segments)  # septal wall
    lv_endo = cyl_lv_endo_fw - box_lv_fw + cyl_lv_endo_sw

    # LV epicardium
    cyl_lv_epi_fw = Ellipsoid(Point(0.0, 0.0, 0.0), R_2, R_2, Z_2, segments=segments)  # free wall
    cyl_lv_epi_sw = Ellipsoid(Point(0.0, 0.0, 0.0), R_2sep, R_2, Z_2, segments=segments)  # septal wall
    lv_epi = cyl_lv_epi_fw - box_lv_fw + cyl_lv_epi_sw

    # RV endocardium
    rv_endo = Ellipsoid(Point(center_rv_x, 0.0, 0.0), R_3, R_3y, Z_3, segments=segments)

    # RV epicardium
    rv_epi = Ellipsoid(Point(center_rv_x, 0.0, 0.0), R_4, R_2, Z_4, segments=segments)

    # Final RV
    box_rv = Box(Point(0., -1.1 * R_2, -1.1 * Z_2),
                 Point(1.1 * R_4, 1.1 * R_2, 1.1 * Z_2))  # box to remove right part of the RV (at x>=0 )
    rv = rv_epi - rv_endo - box_rv

    # Truncation box at base
    box_trunc = Box(Point(-1.1 * (R_2sep + R_4), -1.1 * R_2, h), Point(1.1 * (R_2sep + R_4), 1.1 * R_2, 1.1 * Z_2))

    # Create final domain
    domain = lv_epi + rv - lv_endo - box_trunc
    print('domain biventricular mesh:{}'.format(domain))
    domain = CSGCGALDomain3D(domain)

    # Instantiate and configure the mesh generator.
    # noinspection PyArgumentList
    generator = CSGCGALMeshGenerator3D()
    generator.parameters.update(kwargs.get('mesh_generator_parameters', {}))
    # info(generator.parameters, True)

    # Create final mesh
    return generator.generate(domain)


def compute_biventricular_parameters(independent_parameters):
    """
    Computes geometric parameters that define the ellipsoids that make up
    the simple biventricular geometry as described by Adrián Flores de la Parra.
    This code is a port from the Matlab code written by Bovendeerd and Flores.

    Args:
        independent_parameters: Dolfin parameter set containing the 8 independent
                         parameters that define the geometry as described by
                         Adrian Flores De La Parra.

    Returns:
        A dolfin parameter set with the geometric parameters
        (radii of ellipsoids and truncation height).
    """
    print('Computing biventricular geometric parameters..')

    def compute_V_2(Z_2, Z_1, R_1, R_1sep, h):
        """
        Computes V_2 (epicardial volume) from Z_2 analytically according to Eqs. 2.8 - 2.10.
        """
        f_h2 = h/Z_2   # Eq. 2.8
        k_2 = np.pi/6 * (2 + 3*f_h2 - f_h2**3)   # Eq. 2.8
        R_2 = np.sqrt(Z_2**2 - Z_1**2 + R_1**2)   # Eq. 2.9
        V_2 = k_2 * Z_2 * R_2 * (R_2 + R_2 - R_1 + R_1sep)   # Eq. 2.8 & Eq. 2.10
        return V_2

    def compute_dimensions(R_4, t_RV, x_A, z_A):
        """
        Compute ellipsoidal dimensions for the RV for given R_4.
        """
        R_3 = R_4 - t_RV
        # Compute ellipsoidal coordinate theta for the RV epicardium that corresponds to the attachment point (where LV and RV attach).
        theta_e = np.arcsin(x_A/R_4)
        # Compute corresponding Z_4.
        Z_4 = -z_A / np.cos(theta_e)
        Z_3 = Z_4 - t_RV   # Eq. 2.14

        return R_3, Z_3, Z_4

    def compute_Vrv0(R_3, R_3y, Z_3, R_2sep, R_2, Z_2, R_1, Z_1, h, n_theta, n_phi):
        """
        Computes RV endocardial volume. Ported from the Matlab code by Flores and Bovendeerd.
        """

        def compute_theta_att(R_3, Z_3, Z_2, R_2sep):
            """
            Compute the RV endocardium theta of attachment at y=0.
            """
            err = -1
            dtheta = 0.00001
            theta_att = np.pi/2

            # Search theta_att where LV and RV attach.
            while np.abs(err) > 0.01:
                theta_att += dtheta
                x = -R_3 * np.sin(theta_att)
                z = Z_3 * np.cos(theta_att)
                err = (z/Z_2)**2 + (x/R_2sep)**2 - 1
            return theta_att

        def compute_phi_att(R_3, R_3y, R_2sep, R_2):
            """
            Compute the RV endocardium phi of attachment at z=0.
            """
            err = 1
            dphi=0.00001
            phi_att = 3*np.pi/2

            while np.abs(err) > 0.01:
                phi_att += dphi
                x = -R_3 * np.cos(phi_att)
                y = R_3y * np.sin(phi_att)
                err = (x/R_2sep)**2 + (y/R_2)**2 - 1
            return phi_att

        def obtain_coordinates(n_phi, z_3p, z_2p, t_3, t_2, h, z_att, cutnumber, p_3, R_3, R_3y, Z_3, R_2sep, R_2, Z_2):
            z_att = - z_att # Make z_att positive.
            z_3plane = np.linspace(h*0.99999, (0.001*z_att) - z_att, cutnumber) # Avoid getting out of bounds when interpolating (avoid floating-point arithmetic issues...).

            cos_t3_int_f = interpolate.interp1d(z_3p, np.cos(t_3), kind='cubic', bounds_error=False)
            cos_t2_int_f = interpolate.interp1d(z_2p, np.cos(t_2), kind='cubic', bounds_error=False)
            cos_t3_int = cos_t3_int_f(z_3plane)
            cos_t2_int = cos_t2_int_f(z_3plane)

            t3_int = np.arccos(cos_t3_int)
            t2_int = np.arccos(cos_t2_int)

            x_3d = np.zeros((len(t3_int), len(p_3)))
            y_3d = np.zeros((len(t3_int), len(p_3)))
            z_3d = np.zeros((len(t3_int), len(p_3)))

            x_4d = np.zeros((len(t3_int), len(p_3)))
            y_4d = np.zeros((len(t3_int), len(p_3)))
            z_4d = np.zeros((len(t3_int), len(p_3)))

            for i in range(len(t3_int)-1, -1, -1):
                # Determine phi_att, i.e. RV endocardium phi of attachment.
                err = 1
                d_phi = 0.0001
                phi_C = 3*np.pi/2

                while np.abs(err)>0.001:
                    phi_C += d_phi
                    x_3 = -R_3 * np.cos(phi_C) * np.sin(t3_int[i])
                    y_3 = R_3y * np.sin(phi_C) * np.sin(t3_int[i])
                    x_2 = -R_2sep * np.sin(t2_int[i])
                    y_2 = R_2 * np.sin(t2_int[i])
                    err = (y_3/y_2)**2 + (x_3/x_2)**2 - 1

                p_3min = phi_C
                p_3max = 2*np.pi - phi_C + 2*np.pi
                p_3 = np.linspace(p_3min, p_3max, n_phi)
                x_3d[i,:] = -R_3 * np.cos(p_3) * np.sin(t3_int[i])
                y_3d[i,:] = R_3y * np.sin(p_3) * np.sin(t3_int[i])
                z_3d[i,:] = Z_3 * np.cos(t3_int[i])

                possible_C = 2*np.pi - np.arcsin(y_3/y_2)

                p_4min = possible_C
                p_4max = 2*np.pi - possible_C + 2*np.pi
                p_4 = np.linspace(p_4min, p_4max, n_phi)
                x_4d[i,:] = -R_2sep * np.cos(p_4) * np.sin(t2_int[i])
                y_4d[i,:] = R_2 * np.sin(p_4) * np.sin(t2_int[i])
                z_4d[i,:] = Z_2 * np.cos(t2_int[i])

            return x_3d, y_3d, z_3d, x_4d, y_4d, z_4d

        def calculate_volume(x, y, z):
            volume = 0.
            for i in range(0, len(x[:,0]) - 1):
                for j in range(0, len(x[:,0]) - 1):
                    a = np.array((x[i,j], y[i,j], z[i,j]))
                    b = np.array((x[i,j+1], y[i,j+1], z[i,j+1]))
                    c = np.array((x[i+1,j], y[i+1,j], z[i+1,j]))
                    d = np.array((x[i+1,j+1], y[i+1,j+1], z[i+1,j]))
                    O = np.array((0, 0, z[0,0]))
                    vol_a = np.abs((1/6)*(np.dot(O-a, np.cross(b-a, c-a))))
                    vol_b = np.abs((1/6)*(np.dot(O-a, np.cross(b-d, c-d))))
                    volume += (vol_a + vol_b)
            return volume

        # Compute ellipsoidal parameters ksi_3 and C_3 corresponding to
        # ellipsoid with R_3 and Z_3. Also find theta where z=h.
        if R_3<Z_3:
            ksi_3 = np.arctanh(R_3/Z_3)
            C_3 = np.sqrt(Z_3**2 - R_3**2)
            theta_3 = np.arccos(h/(C_3*np.cosh(ksi_3)))
        else: # R3 is the long axis and Z3 a short axis
            ksi_3 = np.arctanh(Z_3/R_3)
            C_3 = np.sqrt(R_3**2 - Z_3**2)
            theta_3 = np.arccos(h/(C_3*np.sinh(ksi_3)))

        # Find theta_att, i.e. RV endocard theta of attachment (theta_att) at y=0.
        theta_att = compute_theta_att(R_3, Z_3, Z_2, R_2sep)
        # Find phi_att, i.e. RV endocard phi of attachment at z=0.
        phi_att = compute_phi_att(R_3, R_3y, R_2sep, R_2)

        # RV at y=0.
        t_3min = theta_3
        t_3max = theta_att
        t_3 = np.linspace(t_3min, t_3max, n_theta)
        z_3p = Z_3 * np.cos(t_3)

        # RV at z=0.
        p_3min = phi_att
        p_3max = 2*np.pi - phi_att + 2*np.pi
        p_3 = np.linspace(p_3min, p_3max, n_phi)

        # LV epicardium at y=0.
        ksi_2 = np.arctanh(R_2/Z_2)
        C_1 = np.sqrt(Z_1**2 - R_1**2)
        theta_2 = np.arccos(h/(C_1 * np.cosh(ksi_2)))
        t_2min = np.pi
        t_2max = 2*np.pi - theta_2
        t_2 = np.linspace(t_2min, t_2max, n_theta)
        z_2p = Z_2 * np.cos(t_2)

        # z-coordinate of lowest attachment RV endocardium and septal wall.
        z_att = Z_3 * np.cos(theta_att)

        # Obtain coordintates.
        x_3d, y_3d, z_3d, x_4d, y_4d, z_4d = obtain_coordinates(n_phi, z_3p, z_2p, t_3, t_2, h, z_att, 31, p_3, R_3, R_3y, Z_3, R_2sep, R_2, Z_2)

        # Calculate volume
        vol_3 = calculate_volume(x_3d, y_3d, z_3d)
        vol_4 = calculate_volume(x_4d, y_4d, z_4d)
        return vol_3 - vol_4

    # ------------------------------------------------------------------- #
    # Extract independent parameters.
    # ------------------------------------------------------------------- #
    f_R = independent_parameters['f_R']
    f_V = independent_parameters['f_V']
    f_h1 = independent_parameters['f_h1']
    f_T = independent_parameters['f_T']
    f_sep = independent_parameters['f_sep']
    f_VRL = independent_parameters['f_VRL']
    V_lvw = independent_parameters['V_lvw']
    theta_A = independent_parameters['theta_A']

    # Set number of angles for evaluation of points when computing RV endocardial volume.
    n_theta = 36
    n_phi = 31

    # ------------------------------------------------------------------- #
    # Compute ellipsoidal parameters.
    # ------------------------------------------------------------------- #
    # t0=time.time()
    # LV endocardium
    # ------------------------------------------------------------------- #
    V_lv0 = f_V * V_lvw   # Eq. 2.1
    k_1 = np.pi/6 * (2 + 3*f_h1 - f_h1**3)   # Eq. 2.2
    Z_1 = (V_lv0/(k_1*(f_R*(1+f_h1))**2 * (1+f_sep)))**(1/3)   # Eq. 2.6
    h = f_h1 * Z_1   # Eq. 2.3
    R_1 = f_R * (1 + f_h1) * Z_1   # Eq. 2.4
    R_1sep = f_sep * f_R * (1 + f_h1) * Z_1   # Eq. 2.5

    # ------------------------------------------------------------------- #
    # LV epicardium
    # ------------------------------------------------------------------- #
    # Find Z_2 iteratively: search Z_2 so that the resulting volume equals V_2target.
    V_2target = V_lvw * (1 + f_V)   # Eq. 2.7

    # Set search region
    Z_2max = 2*Z_1
    Z_2min = Z_1

    # Compute minimal and maximal volumes of search region.
    V_2max = compute_V_2(Z_2max, Z_1, R_1, R_1sep, h)
    V_2min = compute_V_2(Z_2min, Z_1, R_1, R_1sep, h)
    dV = V_2max - V_2min

    eps_V = 0.001   # Accuracy of volume corresponding to numerically found Z_2

    while dV > eps_V:
        # Narrow down search region.
        Z_2mean = (Z_2max + Z_2min)/2
        V_2mean = compute_V_2(Z_2mean, Z_1, R_1, R_1sep, h)
        if V_2mean > V_2target:
            Z_2max = Z_2mean
            V_2max = V_2mean
        else:
            Z_2min = Z_2mean
            V_2min = V_2mean
        dV = V_2max - V_2min

    Z_2 = (Z_2max + Z_2min)/2   # Note that Z_2max ~ Z_2min ~ Z_2mean
    R_2 = np.sqrt(Z_2 ** 2 - Z_1 ** 2 + R_1 ** 2)  # Eq. 2.9
    R_2sep = R_2 - R_1 + R_1sep   # Eq. 2.10

    # ------------------------------------------------------------------- #
    # RV endocardium & epicardium
    # ------------------------------------------------------------------- #

    t_RV = f_T * (R_2 - R_1) # Thickness of RV wall.
    R_3y = R_2 - t_RV   # Eq. 2.14

    # Compute attachment coordinates from theta_A.
    x_A = - R_2sep * np.sin(theta_A)
    z_A = Z_2 * np.cos(theta_A)

    # Set search region.
    R_4max = (Z_2 + R_2sep)*2
    R_4min = R_2sep + t_RV + 1

    V_rv0target = f_VRL * V_lv0   # Eq. 2.12

    eps_V=0.1 # Percentage of difference with target volume
    dV = 1e10

    while dV > eps_V:
        R_4mean = (R_4max + R_4min) / 2

        R_3mean, Z_3mean = compute_dimensions(R_4mean, t_RV, x_A, z_A)[0:2]

        Vmean = compute_Vrv0(R_3mean, R_3y, Z_3mean, R_2sep, R_2, Z_2, R_1, Z_1, h, n_theta, n_phi)

        if Vmean > V_rv0target:
            R_4max = R_4mean
        else:
            R_4min = R_4mean
        dV = np.abs(V_rv0target - Vmean)/V_rv0target*100

    R_4 = R_4mean
    R_3, Z_3, Z_4 = compute_dimensions(R_4, t_RV, x_A, z_A)

    # Collect the computed values in a parameter set.
    geometry = Parameters('geometry')
    geometry.add('R_1', R_1)
    geometry.add('Z_1', Z_1)
    geometry.add('R_2', R_2)
    geometry.add('Z_2', Z_2)
    geometry.add('R_1sep', R_1sep)
    geometry.add('R_2sep', R_2sep)
    geometry.add('R_3', R_3)
    geometry.add('R_3y', R_3y)
    geometry.add('Z_3', Z_3)
    geometry.add('R_4', R_4)
    geometry.add('Z_4', Z_4)
    geometry.add('h', h)
    # print('Completed in {} s.'.format(time.time()-t0))
    return geometry


def ThickWalledSphereMesh(inner_radius, outer_radius, resolution, **kwargs):
    """
    Tetrahedral mesh of a thick-walled sphere truncated above the xy-plane.

    Args:
        inner_radius: Radius of the inner sphere.
        outer_radius: Radius of the outer sphere.
        resolution: Mesh resolution parameter.
        **kwargs (optional): Arbitrary keyword arguments for the mesh generator.

    Returns:
        :class:`~dolfin.Mesh`
    """
    # Check if segments were given. Use custom default if not.
    segments = int(kwargs.get('segments', 20))

    # Create the inner and outer sphere volumes.
    inner_volume = Sphere(Point(), inner_radius, segments)
    outer_volume = Sphere(Point(), outer_radius, segments)

    # Create a box above the cutoff height and a bit larger than the spheres.
    p1 = Point(1.1*outer_radius, 1.1*outer_radius, 0)
    p2 = Point(-1.1*outer_radius, -1.1*outer_radius, 1.1*outer_radius)
    excess = Box(p1, p2)

    # Instantiate and configure the mesh generator.
    # noinspection PyArgumentList
    generator = CSGCGALMeshGenerator3D()
    generator.parameters['mesh_resolution'] = float(resolution)
    # TODO Implement way to check/configure additional parameters.

    # Combine the domains into a single geometry, mesh, and return.
    domain = CSGCGALDomain3D(outer_volume - inner_volume - excess)
    return generator.generate(domain)



