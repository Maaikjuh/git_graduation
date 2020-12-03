from dolfin import *
import math as math
import numpy as np
import os
import csv

import cvbtk
from cvbtk.mechanics import compute_coordinate_expression
from cvbtk import LeftVentricleGeometry, read_dict_from_csv, save_to_disk, scalar_space_to_vector_space, save_dict_to_csv
from cvbtk import GeoFunc

#define parameters below
meshres = 50 # choose 20, 25, 30, 35, 40, 45, 50, 55, 60, 70 or 80
segments = 20

k_init = 5.e-2
k_end = 1.e-2
steps = 10
div_ischemic = 10.

server = False
project_td = False
project_meshres = 20
project_segments = 20

# file name where the outputs are stored
file_name = 'kerckhoffs_ischemic_div10_LAD_k_1'
# file_name = 'kerckhoffs_k_1'


# for meshres in [20, 30, 40, 50, 60, 70, 80]:

dirout = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/Eikonal_meshes/seg_{}_mesh_{}_{}'.format(segments,meshres, file_name)
dirout_project = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/Eikonal_meshes/seg_{}_mesh_{}_{}'.format(project_segments,project_meshres, file_name)
print('saving output to: ' + dirout)

# mesh that is selected for the calculation
filepath = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/lv_maaike_seg{}_res{}_fibers_mesh.hdf5'.format(segments,meshres)
filepath_project = '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/lv_maaike_seg{}_res{}_fibers_mesh.hdf5'.format(project_segments,project_meshres)

# set to None to simulate healthy activation pattern
ischemic_dir = None # '/mnt/c/Users/Maaike/OneDrive - TU Eindhoven/Graduation_project/meshes/ischemic_meshes/seg_{}_res_{}_a0inf_unchanged_droplet_LAD_2/T0.hdf5'.format(segments,meshres)

# if run on the server, the mesh files are in different folders and the result should be saved to a different folder
if server == True:
    dirout = 'eikonal/seg_{}_mesh_{}_{}'.format(segments,meshres, file_name)
    dirout_project = 'eikonal/seg_{}_mesh_{}_{}'.format(project_segments,project_meshres, file_name)
    print('saving output to: ' + dirout)

    # mesh that is selected for the calculation
    filepath = 'data/lv_maaike_seg{}_res{}_fibers_mesh.hdf5'.format(segments,meshres)
    filepath_project = 'data/lv_maaike_seg{}_res{}_fibers_mesh.hdf5'.format(project_segments,project_meshres)


class EikonalProblem(object):

    def __init__(self, dirout, filepath, ischemia = None, **kwargs):
        self.parameters = self.default_parameters()
        self.parameters.update(kwargs)

        # compute epsilon from the eccentricities (given in leftventricular_meshing.py)
        # e_outer = self.parameters['boundary']['outer_eccentricity']
        # e_inner = self.parameters['boundary']['inner_eccentricity']
        # eps_outer = self.cartesian_to_ellipsoidal(self.parameters['boundary']['focus'], eccentricity = e_outer)['eps']
        # eps_inner = self.cartesian_to_ellipsoidal(self.parameters['boundary']['focus'], eccentricity = e_inner)['eps']

        # self.parameters['boundary']['eps_outer'] = eps_outer
        # self.parameters['boundary']['eps_inner'] = eps_inner

        self.dir_out = dirout
        self.ischemia = ischemia


        #save parameters to csv   
        save_dict_to_csv(self.parameters, os.path.join(self.dir_out, 'inputs.csv'))
        save_dict_to_csv(self.parameters['geometry'], os.path.join(self.dir_out, 'inputs.csv'), write = 'a')
        save_dict_to_csv(self.parameters['root_points'], os.path.join(self.dir_out, 'inputs.csv'), write = 'a')

        #set parameters, quadrature degree should be the same to the loaded fiber vectors
        parameters["form_compiler"]["quadrature_degree"] = 4
        parameters["form_compiler"]["representation"] = "uflacs"

        geometry = LeftVentricleGeometry(meshfile=filepath)
        self.mesh = geometry.mesh()
        self.Q = FunctionSpace(self.mesh, 'Lagrange', 2)
        self.VV = TensorFunctionSpace(self.mesh, "Lagrange", 2)

        #extract fiber vectors
        print('extracting fiber vectors')
        self.ef = geometry.fiber_vectors()[0].to_function(None)
        self.es = geometry.fiber_vectors()[1].to_function(None)
        self.en = geometry.fiber_vectors()[2].to_function(None)

        ## Uncomment to save loaded fiber vectors to xdmf
        # self.V1 = scalar_space_to_vector_space(self.Q)
        # ofile = XDMFFile(mpi_comm_world(), os.path.join(self.dir_out,"ef.xdmf"))
        # ofile.write(project(self.ef,V1))
        self._k_prev = Constant(self.parameters['k_init'])
        self._td_prev = Function(self.Q, name='td_prev')

        self.c = Function(self.Q, name='c')
        self.c.assign(Constant(self.parameters['cm']))

        self.a = Function(self.Q, name='a')
        self.a.assign(Constant(self.parameters['am']))
    
    def eikonal(self):
        prm = self.parameters
        ef = self.ef
        es = self.es
        en = self.en

        k= Constant(self.parameters['k_init']) #2.1e-2 #2.1
        c = self.purkinje_fibers(self.c, 'c')
        # c = self.parameters['cm'] #0.067 #(7.5) 
        # a = self.purkinje_fibers(self.a, 'a')
        a = self.parameters['am'] #(0.38)

        # Neumann boundary condition: g (0) on boundary, f in volume
        f = Constant(prm['f'])
        g = Constant(prm['g'])

        # value of activation time of the initial stimulated region(s)
        td0BC = Constant(prm['td0BC'])

        w = TestFunction(self.Q)
        td = Function(self.Q, name = 'td')


        td00 = TrialFunction(self.Q)
        # unknown initial guess
        td0 = Function(self.Q, name = 'td0')

        def boundary(x, on_boundary):
            prm = self.parameters['root_points']
            geom = self.parameters['geometry']
            focus = geom['focus']

            # ellipsoidal positions of the root points
            phi0 = prm['phi0']
            phi1 = prm['phi1']
            phi2 = prm['phi2']
            phi3 = prm['phi3']
            phi4 = prm['phi4']
            theta0 = prm['theta0']
            theta1 = prm['theta1']
            theta2 = prm['theta2']
            theta3 = prm['theta3']
            theta4 = prm['theta4']

            eps_outer = geom['eps_outer']
            eps_inner = geom['eps_inner']

            r = prm['radius']

            # get cartesian mid points of root points
            x0, y0, z0 = self.ellipsoidal_to_cartesian(focus, eps_outer, theta0, phi0)
            x1, y1, z1 = self.ellipsoidal_to_cartesian(focus, eps_inner, theta1, phi1)
            x2, y2, z2 = self.ellipsoidal_to_cartesian(focus, eps_inner, theta2, phi2)
            x3, y3, z3 = self.ellipsoidal_to_cartesian(focus, eps_inner, theta3, phi3)
            # x4, y4, z4 = self.ellipsoidal_to_cartesian(focus, eps_inner, theta4, phi4)

            # check if a node is within the area of a root point (sphere)
            p0 = (x[0] - x0)**2 + (x[1] - y0)**2 + (x[2] - z0)**2 < r**2
            # p0 = (x[1] - y0)**2 + (x[2] - z0)**2 < r**2
            p1 = (x[0] - x1)**2 + (x[1] - y1)**2 + (x[2] - z1)**2 < r**2
            p2 = (x[0] - x2)**2 + (x[1] - y2)**2 + (x[2] - z2)**2 < r**2
            p3 = (x[0] - x3)**2 + (x[1] - y3)**2 + (x[2] - z3)**2 < r**2
            # p4 = (x[0] - x4)**2 + (x[1] - y4)**2 + (x[2] - z4)**2 < r**2

            # eps = self.cartesian_to_ellipsoidal(focus, x=x)['eps']

            
            # return (p0 or p1 or p3 or p4) and eps <= eps_inner + 0.001 or (p2 and eps >= eps_outer - 0.001)
            return ((p0 or p1 or p2 or p3) and on_boundary) 
            # tol = 1.e-1
            # return p0 and (x[0] > 0)#and on_boundary
            # return (abs(x[1]) < tol and abs(x[2]-2.4) < tol and eps <= 0.3713 + tol) or (abs(x[0]) < tol and abs(x[2]-2.4) < tol and eps <= 0.3713 + tol)
            # return eps <= eps_inner + 0.01

        # define dirichlet boundary conditions -> td0BC for the stimulus locations
        print('Creating boundary conditions')
        # bc = DirichletBC(self.Q, td0BC , boundary, method ='pointwise')
        bc = DirichletBC(self.Q, td0BC , boundary)

        # initialize with poisson solution
        log(INFO , "Solving poisson to initialize") 

        # construct the conductivity tensor M from Mi and Me
        M = outer(ef ,ef)+ a * (outer(es, es) + outer(en, en))

        # initial guess: Poisson equation
        a_in = inner(grad(w), (M * grad(td00))) * k * dx
        L = f*w*dx + g*w*ds

        # solver_linear = "mumps"
        solver_linear = 'mumps'      
        solve(a_in == L, td0, bc, solver_parameters = {"linear_solver": solver_linear})

        # Save initilaization in VTK format
        ofile = XDMFFile(mpi_comm_world(), os.path.join(self.dir_out,"Init_td.xdmf"))
        ofile.write(td0)

        # unit normal to the wavefront, pointing away from depolarized tissue
        p = grad(td)
        
        # Eikonal equation
        a = inner(grad(w), (M * p)) * k * dx + c * sqrt(inner(M * p, p))*w*dx

        F = a-L

        # Jacobian of F: dF/du
        J = derivative(F, td)

        # nonlinear variational problem: Find u (td) in V such that:
        # F(u; w) = 0 for all w in V^
        # solver relies on Jacobian (using Newton's method)
        pde = NonlinearVariationalProblem(F, td , bc , J)

        # create solver
        solver = NonlinearVariationalSolver(pde)
        solver.parameters["newton_solver"]["linear_solver"] = solver_linear
        solver.parameters["newton_solver"]["maximum_iterations"] = 10

        # create xdmf file to save iteratively td solution 
        ofile = XDMFFile(mpi_comm_world(), os.path.join(self.dir_out,"td_solution.xdmf"))

        # assign initial guess (td0) to td
        td.assign(td0)

        # define k_end and the size/number of steps
        # that should be taken to reach k_end
        k_end = self.parameters['k']
        k_init = self.parameters['k_init']
        steps = self.parameters['steps']
        dk = (float(k_end) - float(k_init))/steps

        it = 0

        # first iteration k_init should be used (no dk step)
        dk_step = 0.
        
        # store previous value of the solution td
        self.td_prev = td.vector().get_local()

        # iteratively solve Eikonal equation
        # Newton Solver will not converge if you immediately use the end value for k
        # -> solve for current k, adapt k, use previous solution as new initial guess, solve, etc.
        while float(k) >=  (float(k_end) + 0.0001):
            # assign new value to k
            k.assign(float(self.k_prev) + dk_step)

            # because of the iterative change in steps, k could fall below its end value
            # in that case, assign k_end to k
            if float(k) < k_end:
                k.assign(k_end)

            try:
                print('{}%'.format(round((float(k) - k_init)/(k_end - k_init) * 100,2)))
                print('iteration ', it)
                print('k = ', float(k))

                # solve the system
                solver.solve()

                # succesful solve: normal dk step
                dk_step = dk
                
                print('td max = ', round(max(td.vector()),2))
                
            except RuntimeError as error_detail:
                print('Except RuntimeError: {}'.format(error_detail))
                k.assign(float(self.k_prev) + dk_step/2)
                if float(k) >= k_end:
                    print('Failed to solve. Halving dk and re-attempting...')
                    print('re-attempt with k = ', float(k))
                    td.assign(self.td_prev)
                    try:
                        solver.solve()

                        # succesful solve: half dk step to procceed more slowely
                        dk_step = dk/2
                        print('td max = ', round(max(td.vector()),2))

                    except RuntimeError as error_detail:
                        print('Except RuntimeError: {}'.format(error_detail))
                        print('Failed to solve. Halving dk again and re-attempting...')
                        k.assign(float(self.k_prev) + dk_step/4)
                        print('re-attempt with k = ', float(k))
                        td.assign(self.td_prev)
                        try:
                            solver.solve()

                            # succesful solve: quarter dk step to procceed even more slowely
                            dk_step = dk/4
                            print('td max = ', round(max(td.vector()),2))

                        except RuntimeError as error_detail:
                            # last re-attempt did not succeed (k too low)
                            # close xdmf file and save to hdf5 if last re-attempt did not succeed
                            ofile.close()
                            save_dict_to_csv({'last_successful_k': float(self.k_prev)}, os.path.join(self.dir_out, 'inputs.csv'), write = 'a')
                            with HDF5File(mpi_comm_world(), os.path.join(self.dir_out,'td.hdf5'), 'w') as f:
                                f.write(td, 'td')
                                f.write(self.mesh, 'mesh')
                            print('Except RuntimeError: {}'.format(error_detail))

            # write new td function to xdmf for every iteration
            ofile.write(td, float(it))

            it += 1
            # handle for old k and td
            self.k_prev = k
            self.td_prev = td.vector().get_local()

            # assign new k value (adapts all k's in formulas)
            td.assign(td)

            print('-'*50)
        
        # create self.td to be able to project the td on a different mesh
        self.td = td
        ofile.close()

        # save mesh and solution for td as hdf5 to be used later on in the model
        with HDF5File(mpi_comm_world(), os.path.join(self.dir_out,'td.hdf5'), 'w') as f:
            f.write(td, 'td')
            f.write(self.mesh, 'mesh')

        # define the maximum activation time of the endocardium
        # get tds at the endocardium (eps_inner) and find its max value
        focus = prm['geometry']['focus']
        rastr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+{f})*(x[2]+{f}))".format(f=focus)
        rbstr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-{f})*(x[2]-{f}))".format(f=focus)

        sigmastr="(1./(2.*{f})*({ra}+{rb}))".format(ra=rastr,rb=rbstr,f=focus)
        eps = Expression("acosh({sigma})".format(sigma=sigmastr),degree=3,element=self.Q.ufl_element())

        td_inner_cpp = "(eps <= {eps_endo})? td : 0".format(eps_endo = prm['geometry']['eps_inner'])
        td_inner_exp = Expression(td_inner_cpp,degree=3,element=self.Q.ufl_element(), eps=eps, td=td)
        td_endo = Function(self.Q)
        td_endo.interpolate(td_inner_exp)
        print('Endocard activated in {max_endo}ms, whole LV activated in {max_lv}ms'.format(
            max_endo = round(max(td_endo.vector()), 2), max_lv = round(max(td.vector()), 2)))
    
    @property
    def k_prev(self):
        """
        Return the previous k.
        """
        return self._k_prev

    @k_prev.setter
    def k_prev(self, k):
        self.k_prev.assign(k)

    @property
    def td_prev(self):
        """
        Return the previous td.
        """
        return self._td_prev

    @td_prev.setter
    def td_prev(self, td):
        self.td_prev.vector()[:] = td
        self.td_prev.vector().apply('')
    
    def purkinje_fibers(self,val,str_val):
        focus = self.parameters['geometry']['focus']

        # get eps and phi values of the nodes in the mesh
        rastr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+{f})*(x[2]+{f}))".format(f=focus)
        rbstr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-{f})*(x[2]-{f}))".format(f=focus)

        sigmastr="(1./(2.*{f})*({ra}+{rb}))".format(ra=rastr,rb=rbstr,f=focus)
        thetastr="(1./(2.*{f})*({ra}-{rb}))".format(ra=rastr,rb=rbstr,f=focus)
        
        eps = Expression("acosh({sigma})".format(sigma=sigmastr),degree=3,element=self.Q.ufl_element())
        phi = Expression("atan2(x[1],x[0])",degree=3,element=self.Q.ufl_element())
        theta = Expression('acos({theta})'.format(theta=thetastr),degree=3,element=self.Q.ufl_element())

        theta_func = Function(self.Q)
        theta_func.interpolate(theta)
        theta_min = min(theta_func.vector())

        phi_func = Function(self.Q)
        phi_func.interpolate(phi)

        # # define phi values where LV borders to the RV
        # max_phi = max(phi_func.vector()) - 1/6* np.pi
        # min_phi = max(phi_func.vector()) - 5/6* np.pi

        # define epsilons of endocardium, epicardium and midwall
        eps_inner = self.parameters['geometry']['eps_inner']
        eps_sub_endo = self.parameters['geometry']['eps_sub_endo']
        eps_mid = self.parameters['geometry']['eps_mid']
        eps_outer = self.parameters['geometry']['eps_outer']
        # eps_sub_endo = (eps_mid + eps_inner)/2
        border = 0.05
        eps_border = eps_sub_endo + border
        eps_border = eps_mid

        eps_func = Function(self.Q)
        eps_func.interpolate(eps) 
        eps_outer = max(eps_func.vector())
        eps_inner = min(eps_func.vector())

        # sigval = self.parameters[sigstr]/3
        # define scaling parameters for purkinje system
        if str_val == 'c':
            bulk = self.parameters['cm']
            purk = self.parameters['ce']
        elif str_val == 'a':
            bulk = self.parameters['am']
            purk = self.parameters['ae'] 

        theta_max = 11/24 * math.pi

        # C++ syntax: (a)? b : c
        #   if a == True:
        #       b
        #   else:
        #       c

        # all eps larger than eps_border are outside of the purkinje system -> bulk
        purkinje_layer = "(eps > {eps_sub_endo})? {bulk} : purk".format(
            eps_sub_endo = eps_mid, bulk = bulk)

        # transmural linear transition equation from bulk to purkinje system 
        purk_layer = Expression('{bulk} + ({purk} - {bulk}) * ((eps - {eps_sub_endo})/({eps_endo} - {eps_sub_endo}))'.format(
            bulk = bulk, purk = purk, eps_endo = eps_inner, eps_sub_endo = eps_mid), 
            element=self.Q.ufl_element(), degree=3, eps=eps)

        # linear transitions: transition from the base to the purkinje system and transmural transition 
        purk_trans_base_eps = Expression('{bulk} + ({purk} - {bulk}) * ((eps - {eps_sub_endo})/({eps_endo} - {eps_sub_endo}) * (theta - {theta_min})/({theta_max} - {theta_min}))'.format(
            bulk = bulk, purk = purk, eps_endo = eps_inner, eps_sub_endo = eps_mid, theta_min=theta_min,theta_max=theta_max),
            element=self.Q.ufl_element(), degree=3, eps=eps, theta=theta)

        purk_trans = Expression('(theta > {theta_max})? purk_layer : purk_trans_base_eps'.format(theta_max=theta_max),
            element=self.Q.ufl_element(), degree=3, theta=theta, purk_layer=purk_layer,purk_trans_base_eps=purk_trans_base_eps)
        
        ## Below is a slight variation on the Purkinje system. 
        # # transmural linear transition equation from bulk to purkinje system 
        # purk_trans_eps = Expression('{bulk} + ({purk} - {bulk}) * ((eps - {eps_border})/({eps_sub_endo} - {eps_border}))'.format(
        #     bulk = bulk, purk = purk, eps_border = eps_border, eps_sub_endo = eps_sub_endo), 
        #     element=self.Q.ufl_element(), degree=3, eps=eps)

        # # check if eps is within the transmural transition area, if not, eps is in the purkinje layer
        # purk_trans_layer = Expression('(eps >= {eps_sub_endo} && eps <= {eps_border})? purk_trans_eps : {purk}'.format(
        #     eps_border = eps_border, eps_sub_endo = eps_sub_endo, theta_max = theta_max, purk=purk), 
        #     element=self.Q.ufl_element(), degree=3, purk_trans_eps = purk_trans_eps, eps=eps)

        # # linear transition equation from the base (no purkinje) to the purkinje system
        # purk_trans_base_theta = Expression('{bulk} + ({purk} - {bulk}) * ((theta - {theta_min})/({theta_max} - {theta_min}))'.format(
        #     bulk = bulk, purk = purk, theta_min=theta_min,theta_max=theta_max),
        #     element=self.Q.ufl_element(), degree=3, eps=eps, theta=theta)

        # # linear transitions: transition from the base to the purkinje system and transmural transition 
        # purk_trans_base_eps = Expression('{bulk} + ({purk} - {bulk}) * 0.5 * ((eps - {eps_border})/({eps_sub_endo} - {eps_border}) * (theta - {theta_min})/({theta_max} - {theta_min}))'.format(
        #     bulk = bulk, purk = purk, eps_border = eps_border, eps_sub_endo = eps_sub_endo, theta_min=theta_min,theta_max=theta_max),
        #     element=self.Q.ufl_element(), degree=3, eps=eps, theta=theta)

        # # check if eps is within the transmural transition layer:
        # #   if true, eps is both in the transmural transition as in the transition from base to purkinje
        # #   if false, eps is only within the transition layer from base to purkinje
        # purk_trans_base = Expression('(eps >= {eps_sub_endo} && eps <= {eps_border})? purk_trans_base_eps : purk_trans_base_theta'.format(
        #     eps_border = eps_border, eps_sub_endo = eps_sub_endo),
        #     element=self.Q.ufl_element(), degree=3, eps=eps, purk_trans_base_eps=purk_trans_base_eps, purk_trans_base_theta=purk_trans_base_theta)

        # # check if theta is not within the transition from base to purkinje
        # #   if true (not in transition), theta is only within the transmural transition or completely in the purkinje system
        # #   if false (within transition), theta is either in both the transmural transition and the transition from base to purkinje
        # #       or only within the transition from base to purkinje
        # purk_trans = Expression('(theta > {theta_max})? purk_trans_layer : purk_trans_base'.format(theta_max=theta_max),
        #     element=self.Q.ufl_element(), degree=3, theta=theta, purk_trans_layer=purk_trans_layer,purk_trans_base=purk_trans_base)

        # convert purkinje layer + transitions to Function
        purk = Function(self.Q)
        purk.interpolate(purk_trans)

        # combine bulk and purkinje layer Expressions
        bulk_purk_exp = Expression(purkinje_layer, element=self.Q.ufl_element(), degree=3, eps=eps, purk = purk)
        val.interpolate(bulk_purk_exp)

        if self.ischemia != None:
            print('ischemia')
            T0_file = HDF5File(mpi_comm_world(), self.ischemia, 'r')
            vector_T0 = 'T0/vector_0'
            T0_f = Function(self.Q)
            T0_file.read(T0_f, vector_T0)

            # the wave velocities of the nodes in the ischemic area (defined by T0 <= 60) are dived by a certain factor
            ischemic_sig = "(T0_f <= 60.)? val/{div_isch} : val".format(div_isch = self.parameters['div_ischemic'])

            ischemic_exp = Expression(ischemic_sig, element=self.Q.ufl_element(), degree=3, T0_f = T0_f, val = val)
            val = Function(self.Q)
            val.interpolate(ischemic_exp)

        # write values to xdmf for easy viewing. 
        ofile = XDMFFile(mpi_comm_world(), os.path.join(self.dir_out,"{}.xdmf".format(str_val)))
        ofile.write(val)

        return val
    
    def cartesian_to_ellipsoidal(self, focus, x = [0,0,0], eccentricity = None):
        if eccentricity != None:
            x[0] = focus*np.sqrt(1-eccentricity**2)/eccentricity
            x[1] = 0
            x[2] = 0

        ra = sqrt(x[0]**2+x[1]**2+(x[2]+focus)**2)
        rb = sqrt(x[0]**2+x[1]**2+(x[2]-focus)**2)

        tau= (1./(2.*focus)*(ra-rb))
        sigma=(1./(2.*focus)*(ra+rb))

        expressions_dict = {"phi": math.atan2(x[1],x[0]),
                            "eps": math.acosh(sigma),
                            "theta": math.acos(tau)} 
        return expressions_dict

    def ellipsoidal_to_cartesian(self,focus, eps,theta,phi):
        x = focus * math.sinh(eps) * math.sin(theta) * math.cos(phi)
        y = focus * math.sinh(eps) * math.sin(theta) * math.sin(phi)
        z = focus * math.cosh(eps) * math.cos(theta) 

        return x, y, z

    @staticmethod
    def default_parameters():
        prm = Parameters('eikonal')
        prm.add('k_init', 2.e-2) #Initialisation value of the diffusion constant 
        prm.add('k', 5.e-3) #Diffusion constant -> influence of wavefront curvature on wave velocity. ker03b + ker03a: 2.1e-3

        # c = theta -> wave velocity in the myofiber direction
        prm.add('cm', 0.067) #wave velocity in the myofiber direction in the myocardium. ker03b: 0.075
        prm.add('am', 0.38) #Anisotropy ratio (c_parallel / c_perpendicular to the myofibers) in the myocardium. ker03b: 1/2.5
        prm.add('ce', 0.4) #wave velocity in the myofiber direction in the endocardium. ker03b: 0.13
        # prm.add('ae', 1/1.5) #Anisotropy ratio (c_parallel / c_perpendicular to the myofibers) in the myocardium. ker03b: 1/1.5 (not used in ker03a)
        prm.add('steps', 10)
        # prm.add('sig_fac_purk', 2.)
        # prm.add('sig_il', (2e-3)) #(2e-3)*2
        # prm.add('sig_it', (4.16e-4)) #(4.16e-4)*2
        # prm.add('sig_el', (2.5e-3)) #(2.5e-3)*2
        # prm.add('sig_et', (1.25e-3)) #(1.25e-3)*2
        prm.add('div_ischemic', 2.) #divide c in the ischemic area to simulate slower depolarization velocities

        prm.add('f', (1))
        prm.add('g', (0))
        prm.add('td0BC', 0.0)

        geometry_prm = Parameters('geometry')
        geometry_prm.add('eps_purk', 0.4)
        geometry_prm.add('focus', 4.3)
        geometry_prm.add('outer_eccentricity', 0.807075)
        geometry_prm.add('inner_eccentricity', 0.934819)
        geometry_prm.add('eps_outer', 0.6784)
        geometry_prm.add('eps_mid', 0.5305)
        geometry_prm.add('eps_sub_endo', 0.41787)
        geometry_prm.add('eps_inner', 0.3713)

        prm.add(geometry_prm)

        # locations of the root points (td = 0)
        boundary_prm = Parameters('root_points')
        boundary_prm.add('phi0', 0.)
        boundary_prm.add('phi1', 0.) #-1/8*math.pi
        boundary_prm.add('phi2', -1/2*math.pi)    #1/2*math.pi
        boundary_prm.add('phi3', math.pi) 
        boundary_prm.add('phi4', -3/4*math.pi)   
        boundary_prm.add('theta0', 2/3*math.pi)#1/2*math.pi
        boundary_prm.add('theta1', 2/3*math.pi) #9/16*math.pi
        boundary_prm.add('theta2', 2/3*math.pi) #9/16*math.pi
        boundary_prm.add('theta3', 2/3*math.pi)
        boundary_prm.add('theta4', 1.2854)
        boundary_prm.add('radius', .5)   
        boundary_prm.add('tol', 1.e-1)
        boundary_prm.add('tol_theta', 1.e-1)

        prm.add(boundary_prm)
        return prm

    def project(self, filepath, dirout):
        print('projecting onto {}'.format(os.path.split(filepath)[1]))
        mesh = Mesh()

        openfile = HDF5File(mpi_comm_world(), filepath, 'r')
        openfile.read(mesh, 'mesh', False)
        V = FunctionSpace(mesh, "Lagrange", 2)

        self.td.set_allow_extrapolation(True)

        td_project = project(self.td, V)

        td_project.rename('td','td')

        # write values to xdmf for easy viewing. 
        ofile = XDMFFile(mpi_comm_world(), os.path.join(dir_out,"td_solution.xdmf"))
        ofile.write(td_project)
        # file = File(os.path.join(dirout,"td_solution.pvd"))
        # file << td_project

        # save mesh and solution for td as hdf5 to be used later on in the model
        with HDF5File(mpi_comm_world(), os.path.join(dirout,'td.hdf5'), 'w') as f:
            f.write(td_project, 'td')
            f.write(mesh, 'mesh')

if __name__ == '__main__':
    inputs = {  'k': k_end,
                'k_init': k_init,
                'steps': steps,
                'div_ischemic': div_ischemic}

    # problem = EikonalProblem(dirout, filepath, **inputs)
    problem = EikonalProblem(dirout, filepath, ischemia =  ischemic_dir, **inputs)
    problem.eikonal()

    if project_td == True:
        problem.project(filepath_project, dirout_project)


