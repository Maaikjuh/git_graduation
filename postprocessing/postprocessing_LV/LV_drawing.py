# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 16:46:11 2018

@author: Hermans
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

R_1 = 1.86
R_2 = 3.42
C = 4.85
h = 2.1

Z_1 = np.sqrt(C**2 + R_1**2)
Z_2 = np.sqrt(C**2 + R_2**2)

t = np.linspace(1/2*np.pi, 5/2*np.pi, 1000)
x_1 = R_1*np.cos(t)
y_1 = Z_1*np.sin(t)
x_1 = x_1[y_1 <= h]
y_1 = y_1[y_1 <= h]

x_2 = R_2*np.cos(t)
y_2 = Z_2*np.sin(t)
x_2 = x_2[y_2 <= h]
y_2 = y_2[y_2 <= h]

plt.figure(figsize=(8, 8), dpi=200)
plt.plot(x_1, y_1, 'k--')
plt.plot(x_2, y_2, 'k')
plt.gca().set_aspect('equal', 'datalim')
#plt.grid('on')
plt.xlabel('x (cm)')
plt.ylabel('z (cm)')
a = 1.2
#plt.axis([-R_2*a, R_2*a, -Z_2*a, h*a])
plt.axis('off')

offset = 0.1

a = 2.1
plt.annotate(
    '$x$', xy=(offset, offset), xycoords='data',
    xytext=(a*-2, a*-1.2), textcoords='data', 
    arrowprops={'arrowstyle': '<-'})

plt.annotate(
    '$y$', xy=(0, 0), xycoords='data',
    xytext=(R_2*1.5, -0.095), textcoords='data', 
    arrowprops={'arrowstyle': '<-'})

plt.annotate(
    '$z$', xy=(0, 0), xycoords='data',
    xytext=(-0.08, h*2), textcoords='data', 
    arrowprops={'arrowstyle': '<-'})

x_drift = 4.
plt.annotate(
    '', xy=(x_drift, -offset), xycoords='data',
    xytext=(x_drift, h+offset), textcoords='data',
    arrowprops={'arrowstyle': '<->'})
plt.annotate(
    '$h$', xy=(x_drift, h/2), xycoords='data',
    xytext=(5, 0), textcoords='offset points')

plt.annotate(
    '', xy=(x_drift, offset), xycoords='data',
    xytext=(x_drift, -C-offset), textcoords='data', 
    arrowprops={'arrowstyle': '<->'})

plt.annotate(
    '$C$', xy=(x_drift, -C/2), xycoords='data',
    xytext=(5, 0), textcoords='offset points')


f = 0.2
r_1 = abs(x_1[0])
a_1 = r_1*f
r_2 = abs(x_2[0])
a_2 = r_2*f
r_3 = R_2
a_3 = f*r_3
x0_1 = 0
y0_1 = h
x0_2 = 0
y0_2 = h

xe_1 = x0_1 + r_1*np.cos(t)
ye_1 = y0_1 + a_1*np.sin(t)
xe_2 = x0_2 + r_2*np.cos(t)
ye_2 = y0_2 + a_2*np.sin(t)

plt.plot(xe_1, ye_1, 'k')
plt.plot(xe_2, ye_2, 'k')
t = np.linspace(-np.pi, 0, 500)
plt.plot(r_3*np.cos(t), a_3*np.sin(t), 'k')
t = np.linspace(0, np.pi, 500)
plt.plot(r_3*np.cos(t), a_3*np.sin(t), 'k--')
plt.hlines(0, -R_2*1.2, R_2*1.2)
plt.vlines(0, -Z_2*1.2, h*1.8)
plt.hlines(h, 0, R_2*1.5)
plt.scatter(0, h, color='k')
plt.hlines(-C, 0, R_2*1.5)
plt.scatter(0, -C, color='k')
plt.scatter(0, 0, color='k')
plt.text(-0.5, 0.1, '$\it{O}$')

plt.savefig('lv_geometry.png', dpi=300, bbox_inches="tight")