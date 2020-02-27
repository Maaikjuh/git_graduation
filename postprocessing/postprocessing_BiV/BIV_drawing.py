# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:16:02 2018

@author: Hermans
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

def ellips(a, b, t):
    x = a*np.cos(t)
    y = b*np.sin(t)
    return x, y

def cutoff(x, y, h):
    x = x[y<=h]
    y = y[y<=h]
    return x, y

def compute_x_attach(a1, a2, b1, b2):
    c = (b1/a1)**2 - (b2/a2)**2
    d = b1**2 - b2**2
    x = -np.sqrt(d/c)
    return x

def compute_x_att_endo(z, **geometry_parameters):
    """
    Computes the x-coordinate where the LV epicardium and RV endocardium attach.
    NOTE: only works when center of LV is (0, 0, 0).
    :return: x_att_endo
    """

    # Extract geometric parameters.
    R_2 = geometry_parameters['R_2']
    R_2sep = geometry_parameters['R_2sep']
    R_3 = geometry_parameters['R_3']
    R_3y = geometry_parameters['R_3y']
    Z_2 = geometry_parameters['Z_2']
    Z_3 = geometry_parameters['Z_3']

    # Compute d_1 and d_2
    d_1 = 1 - (z / Z_2) ** 2
    d_2 = 1 - (z / Z_3) ** 2

    # Solve abc forumula.
    center_rv_x = 0. # center of RV is at origin.
    a = 1 / (R_3 ** 2) - (R_2 / (R_2sep * R_3y)) ** 2
    b = -2 * center_rv_x / (R_3 ** 2)
    c = (R_2 / R_3y) ** 2 * d_1 - d_2 + (center_rv_x / R_3) ** 2

    nom1 = -b + np.sqrt(b ** 2 - 4 * a * c)
    nom2 = -b - np.sqrt(b ** 2 - 4 * a * c)
    den = 2 * a

    # Return lowest x value
    return np.min((nom1 / den, nom2 / den))

R_1 = 2.02
R_1sep = 1.42
R_2 = 3.56
R_2sep = 2.96
R_3 = 5.73
R_3y = 3.06
R_4 = 6.24
Z_1 = 5.25
Z_2 = 6.02
Z_3 = 4.98
Z_4 = 5.49
h = 2.10

n = 500
lw = 8

# short axis view.
plt.figure(figsize=(20, 24))
ax1=plt.subplot(121)
plt.axis('equal')
plt.axis('off')

# LV free wall
t_lvfw = np.linspace(-np.pi/2, np.pi/2, n)
x_endo, y_endo = ellips(R_1, R_1, t_lvfw)
x_epi, y_epi = ellips(R_2, R_2, t_lvfw)

plt.plot(x_endo, y_endo, color='C3', linewidth=lw)
plt.plot(x_epi, y_epi, color='C3', linewidth=lw)

# LV septal wall
t_lvsep = np.linspace(np.pi/2, 1.5*np.pi, n)
x_endo, y_endo = ellips(R_1sep, R_1, t_lvsep)
x_epi, y_epi = ellips(R_2sep, R_2, t_lvsep)

plt.plot(x_endo, y_endo, color='C2', linewidth=lw)
plt.plot(x_epi, y_epi, color='C2', linewidth=lw)

# RV free wall
x = compute_x_att_endo(0, R_2=R_2, R_2sep=R_2sep, R_3=R_3, R_3y=R_3y, Z_2=Z_2, Z_3=Z_3)
t_min = np.arccos(x/R_3)
t_max = np.arccos(-x/R_3)+np.pi
x_endo, y_endo = ellips(R_3, R_3y, np.linspace(t_min, t_max, n))

t_rv = np.linspace(np.pi/2, 1.5*np.pi, n)
x_epi, y_epi = ellips(R_4, R_2, t_rv)

plt.plot(x_endo, y_endo, color='C0', linewidth=lw)
plt.plot(x_epi, y_epi, color='C0', linewidth=lw)

y_offset = -0.1
x_offset = -0.13
fontsize = 15
plt.scatter(0, 0, color='k')
plt.annotate(
    '$x$', xy=(0., 0.), xycoords='data', fontsize=fontsize,
    xytext=(1, y_offset), textcoords='data', 
    arrowprops={'arrowstyle': '<-', 'lw': 2})
plt.annotate(
    '$y$', xy=(0., 0.), xycoords='data', fontsize=fontsize,
    xytext=(x_offset, 1), textcoords='data', 
    arrowprops={'arrowstyle': '<-', 'lw': 2})
plt.text(-0.35, -0.35, '$\it{O}$', fontsize=fontsize)

# Long axis
plt.subplot(122, sharex=ax1)
plt.axis('equal')
plt.axis('off')

# LV free wall
t_lvfw = np.linspace(-np.pi/2, np.pi/2, n)
x_endo, y_endo = ellips(R_1, Z_1, t_lvfw)
x_epi, y_epi = ellips(R_2, Z_2, t_lvfw)
x_mid, y_mid = ellips((R_1+R_2)/2, (Z_1+Z_2)/2, t_lvfw)

# Cut off
x_endo, y_endo = cutoff(x_endo, y_endo, h)
x_epi, y_epi = cutoff(x_epi, y_epi, h)
x_mid, y_mid = cutoff(x_mid, y_mid, h)

plt.plot(x_endo, y_endo, color='C3', linewidth=lw)
plt.plot(x_epi, y_epi, color='C3', linewidth=lw)
plt.plot(x_mid, y_mid, '--', color='C3', linewidth=lw/2)

# LV septal wall
t_lvsep = np.linspace(np.pi/2, 1.5*np.pi, n)
x_endo, y_endo = ellips(R_1sep, Z_1, t_lvsep)
x_epi, y_epi = ellips(R_2sep, Z_2, t_lvsep)

# Cut off
x_endo, y_endo = cutoff(x_endo, y_endo, h)
x_epi, y_epi = cutoff(x_epi, y_epi, h)

plt.plot(x_endo, y_endo, color='C2', linewidth=lw)
plt.plot(x_epi, y_epi, color='C2', linewidth=lw)


# RV free wall
xa_endo = compute_x_attach(R_2sep, R_3, Z_2, Z_3)
xa_epi = compute_x_attach(R_2sep, R_4, Z_2, Z_4)

t_rv_endo = np.linspace(np.pi/2, -np.arccos(xa_endo/R_3)+2*np.pi, n)
t_rv_epi = np.linspace(np.pi/2, -np.arccos(xa_epi/R_4)+2*np.pi, n)

x_endo, y_endo = ellips(R_3, Z_3, t_rv_endo)
x_epi, y_epi = ellips(R_4, Z_4, t_rv_epi)

# Cut off
x_endo, y_endo = cutoff(x_endo, y_endo, h)
x_epi, y_epi = cutoff(x_epi, y_epi, h)

plt.plot(x_endo, y_endo, color='C0', linewidth=lw)
plt.plot(x_epi, y_epi, color='C0', linewidth=lw)

y_offset = -0.1
x_offset = -0.1
fontsize = 15
plt.scatter(0, 0, color='k')
plt.annotate(
    '$x$', xy=(0., 0.), xycoords='data', fontsize=fontsize,
    xytext=(1, y_offset), textcoords='data', 
    arrowprops={'arrowstyle': '<-', 'lw': 2})
plt.annotate(
    '$z$', xy=(0., 0.), xycoords='data', fontsize=fontsize,
    xytext=(x_offset, 1), textcoords='data', 
    arrowprops={'arrowstyle': '<-', 'lw': 2})
plt.text(-0.35, -0.35, '$\it{O}$', fontsize=fontsize)


