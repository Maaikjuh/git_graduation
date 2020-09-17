#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:26:08 2020

@author: maaike
"""
from dolfin import *
import math
import os
import csv

import cvbtk
from cvbtk import LeftVentricleGeometry, read_dict_from_csv, save_to_disk, scalar_space_to_vector_space, save_dict_to_csv
from cvbtk import GeoFunc

project_mesh = True

a0_infarct = 0.4
Ta0 = 140.0
border = True

segments = 20
resolution = 30

segments_project = 20
resolution_project = 20

dirout = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/meshes/ischemic_meshes/seg_{seg}_res_{res}_a0inf_{a0}_droplet_big_border'.format(
    seg = int(segments), res = int(resolution), a0 = 'unchanged') #int(a0_infarct))
dirout_project = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/meshes/ischemic_meshes/seg_{seg}_res_{res}_a0inf_{a0}_droplet_big_border'.format(
    seg = int(segments_project), res = int(resolution_project), a0 = 'unchanged') #int(a0_infarct) # 'unchanged'

meshfile = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/meshes/lv_maaike_seg{seg}_res{res}_fibers_mesh.hdf5'.format(
    seg = int(segments), res = int(resolution))
meshfile_project = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/meshes/lv_maaike_seg{seg}_res{res}_fibers_mesh.hdf5'.format(
    seg = int(segments_project), res = int(resolution_project))

def compute_coordinate_expression(degree,element,var,focus):
    """
    19-03 Maaike
    Expression for ellipsoidal coordinates (alternative definition)
    https://en.wikipedia.org/wiki/Prolate_spheroidal_coordinates

    Args:
        var : string of ellipsoidal coordinate to be calculated
        V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
        focus: Distance from the origin to the shared focus points.

    Returns:
        Expression for ellipsoidal coordinate
    
    TODO   get **kwargs working 
            make dictionairy with infarct parameters in main to be passed 
    """

    rastr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+{f})*(x[2]+{f}))".format(f=focus)
    rbstr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-{f})*(x[2]-{f}))".format(f=focus)

    taustr= "(1./(2.*{f})*({ra}-{rb}))".format(f=focus,ra=rastr,rb=rbstr)
    sigmastr="(1./(2.*{f})*({ra}+{rb}))".format(ra=rastr,rb=rbstr,f=focus)

    expressions_dict = {"phi": "atan2(x[1],x[0])",
                        "eps": "acosh({sigma})".format(sigma=sigmastr),
                        "theta": "acos({tau})".format(tau=taustr)} 
    return Expression(expressions_dict[var],degree=degree,element=element)
    
class ischemicArea(object):
    
    def __init__(self, dirout, meshfile, **kwargs):
        self.parameters = self.default_parameters()
        self.parameters.update(kwargs)     
        self.dir_out = dirout

        self.infarct_size = 0.
            
        #save parameters to csv   
        save_dict_to_csv(self.parameters, os.path.join(self.dir_out, 'inputs.csv'))

        #set parameters, quadrature degree should be the same to the loaded fiber vectors
        parameters["form_compiler"]["quadrature_degree"] = 4
        parameters["form_compiler"]["representation"] = "uflacs"
        
        geometry = LeftVentricleGeometry(meshfile=meshfile)
        self.mesh = geometry.mesh()
        self.Q = FunctionSpace(self.mesh, 'Lagrange', 2)   
        
        self.T0 = Function(self.Q, name = 'T0')
        self.T0.assign(Constant(self.parameters['Ta0']))

        self.a0 = Function(self.Q, name = 'a0')
        self.a0.assign(Constant(1))
        
    def compute_coordinates(self):
        focus = self.parameters['focus']
        # calculate spherical nodal coordinates of the mesh
        phi = compute_coordinate_expression(3, self.Q.ufl_element(),'phi',focus)
        theta = compute_coordinate_expression(3, self.Q.ufl_element(),'theta',focus)
        eps = compute_coordinate_expression(3, self.Q.ufl_element(),'eps',focus)
        
        return phi, theta, eps

    def create_ischemic_area(self):
        prm = self.parameters
        if prm['border'] == True:
            Ta0_exp, a0_exp  = self.ischemic_droplet_border()
        else:
            Ta0_exp, a0_exp  = self.ischemic_droplet()

        self.T0.interpolate(Ta0_exp)
        self.a0.interpolate(a0_exp)

        self.T0.rename('T0','T0')
        self.a0.rename('a0','a0')

        file = XDMFFile(os.path.join(self.dir_out, 'T0.xdmf'))
        file.write(self.T0)

        file = XDMFFile(os.path.join(self.dir_out, 'a0.xdmf'))
        file.write(self.a0)

        # save mesh and solution for td as hdf5 to be used later on in the model
        with HDF5File(mpi_comm_world(), os.path.join(self.dir_out,'T0.hdf5'), 'w') as f:
            f.write(self.T0, 'T0')
            f.write(self.mesh, 'mesh')
        f.close()

        with HDF5File(mpi_comm_world(), os.path.join(self.dir_out,'a0.hdf5'), 'w') as f:
            f.write(self.a0, 'a0')
            f.write(self.mesh, 'mesh')
        f.close()

        # get volume of the total mesh (with infarct)
        tot_volume = assemble(Constant(1)*dx(domain=self.mesh))       

        # get infarcted volume (Ta0 below 60)
        self.infarct_volume = assemble(conditional(lt(self.T0, 100), 1., 0.)*dx(domain=self.mesh), 
            form_compiler_parameters={'quadrature_degree': 2}) 
        inf_volume_perc = self.infarct_volume/tot_volume*100
        inf_volume_perc = round(inf_volume_perc,1)
        print("infarct volume: {}%".format(inf_volume_perc))

        save_dict_to_csv({'infarct volume (%)': inf_volume_perc}, 
            os.path.join(self.dir_out, 'inputs.csv'), write = 'a')
        
    
    def ischemic_droplet(self):
        prm = self.parameters
        
        phi, theta, eps = self.compute_coordinates()
        
        # phi slope from zero at theta min and max at the apex (if theta_max is pi),
        # because all lines coincide at the apex, a droplet shape is formed by the linear slope
        slope = (prm['phi_max'])/(prm['theta_max']-(prm['theta_min']))
        #expression for the phi values for the right side of the droplet shape 
        drop_exp = Expression("{slope}*(theta-({theta_min}))".format(
            slope=slope,theta_min=prm['theta_min']), degree=3, theta=theta)

        #check if phi is smaller than the right side of the droplet and bigger than the left side
        cpp_exp_phi = "fabs(phi-{phi0}) <= drop_exp && fabs(phi-{phi0}) >=-1*(drop_exp)".format(
            phi0=prm['phi0'])
        
        #check if theta is within the specified theta range
        cpp_exp_theta = "theta> ({thetamin} ) && theta < ({thetamax})".format(
            thetamin=prm['theta_min'], thetamax=prm['theta_max'])
        
        #check if epsilon is greater than the smallest specified ellipsoid
        cpp_exp_eps = "eps >= {epsmin}".format(
            epsmin=prm['epsmin'])

        # if in infarct area: T0 = specified Ta0 for the infarct
        # else: T0 = Ta0
        cpp_exp_Ta0 = "({exp_phi} && {exp_theta} && {exp_xi})? {Ta0_infarct} : {Ta0}".format(
            Ta0_infarct=prm['Ta0_infarct'],Ta0=prm['Ta0'], 
            exp_phi=cpp_exp_phi, exp_theta=cpp_exp_theta, exp_xi=cpp_exp_eps)

        cpp_exp_a0 = "({exp_phi} && {exp_theta} && {exp_xi})? {a0_infarct} : {a0}".format(
            a0_infarct = prm['a0_infarct'], a0 = prm['a0'], 
            exp_phi=cpp_exp_phi, exp_theta=cpp_exp_theta, exp_xi=cpp_exp_eps)

        # Ta0_exp = Expression(cpp_exp_Ta0, element=Q.ufl_element(), phi=phi, theta=theta, xi=xi,drop_exp=drop_exp)
        Ta0_exp = Expression(cpp_exp_Ta0, element=self.Q.ufl_element(), 
            phi=phi, theta=theta, eps=eps,drop_exp=drop_exp)

        a0_exp = Expression(cpp_exp_a0, element=self.Q.ufl_element(), 
            phi=phi, theta=theta, eps=eps,drop_exp=drop_exp)
        return Ta0_exp, a0_exp
        
        
    def ischemic_droplet_border(self):
        prm = self.parameters
        phi, theta, eps = self.compute_coordinates()

        # inner infarct area (no active stress at all)
        min_theta = 1/2*pi + 1/15*math.pi
        max_theta = math.pi -1/5*math.pi
        max_phi = 1/3*math.pi

        # first infarct border
        min_theta_border = 1/2*math.pi-1/12*math.pi
        max_theta_border = math.pi
        max_phi_border = max_phi+1/4*math.pi

        # second infarct border (last)
        min_theta_border_2 = 1/5*math.pi
        max_phi_border_2 = max_phi_border +1/3*math.pi

        slope = max_phi/(math.pi-min_theta)
        slope_border = max_phi_border/(math.pi-min_theta_border)
        slope_border_2 = max_phi_border_2/(math.pi-min_theta_border_2)

        drop_exp = Expression("{slope}*(theta-({theta_min}))".format(slope=slope,theta_min=min_theta), degree=3, theta=theta)
        drop_exp_border = Expression("{slope}*(theta-({theta_min}))".format(slope=slope_border,theta_min=min_theta_border), degree=3, theta=theta)
        drop_exp_border_2 = Expression("{slope}*(theta-({theta_min}))".format(slope=slope_border_2,theta_min=min_theta_border_2), degree=3, theta=theta)

        cpp_exp_phi = "fabs(phi-{phi0}) <=drop_exp && fabs(phi-{phi0}) >=-1*(drop_exp)".format(phi0=prm['phi0'])
        cpp_exp_phi_border = "fabs(phi-{phi0}) <=drop_exp_border && fabs(phi-{phi0}) >=-1*(drop_exp_border)".format(phi0=prm['phi0'])
        cpp_exp_phi_border_2 = "fabs(phi-{phi0}) <=drop_exp_border_2 && fabs(phi-{phi0}) >=-1*(drop_exp_border_2)".format(phi0=prm['phi0'])
    
        cpp_exp_theta = "theta> ({thetamin} ) && theta < ({thetamax})".format(
            thetamin=min_theta, thetamax=max_theta)
        cpp_exp_theta_border = "theta> ({thetamin} ) && theta < ({thetamax})".format(
            thetamin=min_theta_border, thetamax=max_theta_border)
        cpp_exp_theta_border_2 = "theta> ({thetamin} ) && theta < ({thetamax})".format(
            thetamin=min_theta_border_2, thetamax=max_theta_border)
        
        Ta0_infarct = 0.
        diff_Ta0_inf = prm['Ta0'] - Ta0_infarct
        Ta0_border_2 = "({exp_phi} && {exp_theta})? {Ta0_infarct} : {Ta0}".format(
            Ta0_infarct= Ta0_infarct + diff_Ta0_inf * 2/3,
            Ta0=prm['Ta0'], exp_phi=cpp_exp_phi_border_2, exp_theta=cpp_exp_theta_border_2)
        Ta0_border = "({exp_phi} && {exp_theta})? {Ta0_infarct} : {border_2}".format(
            Ta0_infarct=Ta0_infarct + diff_Ta0_inf * 1/3,
            border_2=Ta0_border_2, exp_phi=cpp_exp_phi_border, exp_theta=cpp_exp_theta_border)
        Ta0_infarct = "({exp_phi} && {exp_theta})? {Ta0_infarct} : {border}".format(
            Ta0_infarct=Ta0_infarct,border=Ta0_border, exp_phi=cpp_exp_phi, exp_theta=cpp_exp_theta)

        diff_a0_inf = prm['a0_infarct'] - prm['a0']
        a0_border_2 = "({exp_phi} && {exp_theta})? {a0_infarct} : {a0}".format(
            a0_infarct = prm['a0_infarct'] - diff_a0_inf * 2/3, 
            a0=prm['a0'], exp_phi=cpp_exp_phi_border_2, exp_theta=cpp_exp_theta_border_2)
        a0_border = "({exp_phi} && {exp_theta})? {a0_infarct} : {border_2}".format(
            a0_infarct=prm['a0_infarct'] - diff_a0_inf * 1/3, 
            border_2=a0_border_2, exp_phi=cpp_exp_phi_border, exp_theta=cpp_exp_theta_border)
        a0_infarct = "({exp_phi} && {exp_theta})? {a0_infarct} : {border}".format(
            a0_infarct=prm['a0_infarct'],border=a0_border, exp_phi=cpp_exp_phi, exp_theta=cpp_exp_theta)

        Ta0_exp = Expression(Ta0_infarct, element=self.Q.ufl_element(), 
            phi=phi, theta=theta, drop_exp=drop_exp,drop_exp_border=drop_exp_border,drop_exp_border_2=drop_exp_border_2)

        a0_exp = Expression(a0_infarct, element=self.Q.ufl_element(), 
            phi=phi, theta=theta, drop_exp=drop_exp,drop_exp_border=drop_exp_border,drop_exp_border_2=drop_exp_border_2)
      
        return Ta0_exp, a0_exp

    @staticmethod
    def default_parameters():
        prm = Parameters('ischemic_area')
        prm.add('Ta0', 140.)
        prm.add('Ta0_infarct', 20.)

        prm.add('a0', 0.4)
        prm.add('a0_infarct', 4.)

        prm.add('focus', 4.3)
        prm.add('phi_min', 0.)
        prm.add('phi_max', 1.885)
        prm.add('theta_min', 1.5708)
        prm.add('theta_max', 3.1416)
        prm.add('epsmin', 0.)
        prm.add('phi0', 0.)
        
        prm.add('border', False)
        
        return prm

    def project(self, filepath, dirout):
        print('projecting onto {}'.format(os.path.split(filepath)[1]))

        mesh = Mesh()
        openfile = HDF5File(mpi_comm_world(), filepath, 'r')
        openfile.read(mesh, 'mesh', False)
        V = FunctionSpace(mesh, "Lagrange", 2)  

        self.T0.set_allow_extrapolation(True)
        T0_project = project(self.T0, V)   
        T0_project.rename('T0','T0')   

        self.a0.set_allow_extrapolation(True)
        a0_project = project(self.a0, V)   
        a0_project.rename('a0','a0')

        file = XDMFFile(os.path.join(dirout, 'T0.xdmf'))
        file.write(T0_project)

        file = XDMFFile(os.path.join(dirout, 'a0.xdmf'))
        file.write(a0_project)

        save_dict_to_csv(self.parameters, os.path.join(dirout, 'inputs.csv'))

        tot_volume = assemble(Constant(1)*dx(domain=mesh))
        # get infarcted volume (Ta0 below 100)
        self.infarct_volume = assemble(conditional(lt(T0_project, 100), 1., 0.)*dx(domain=mesh), 
            form_compiler_parameters={'quadrature_degree': 2}) 
        inf_volume_perc = self.infarct_volume/tot_volume*100
        inf_volume_perc = round(inf_volume_perc,1)
        print("infarct volume projected: {}%".format(inf_volume_perc))

        save_dict_to_csv({'infarct volume (%)': inf_volume_perc}, 
            os.path.join(dirout, 'inputs.csv'), write = 'a')

        # save mesh and solution for td as hdf5 to be used later on in the model
        with HDF5File(mpi_comm_world(), os.path.join(dirout,'T0.hdf5'), 'w') as f:
            f.write(T0_project, 'T0')
            f.write(mesh, 'mesh')
        f.close()
        with HDF5File(mpi_comm_world(), os.path.join(dirout,'a0.hdf5'), 'w') as f:
            f.write(a0_project, 'a0')
            f.write(mesh, 'mesh')
        f.close()

if __name__ == '__main__':
    inputs = {'a0_infarct': a0_infarct,
            'Ta0': Ta0,
            'border': border}
    
    ischemic = ischemicArea(dirout, meshfile, **inputs)
    ischemic.create_ischemic_area()
    
    if project_mesh == True:
        ischemic.project(meshfile_project, dirout_project)