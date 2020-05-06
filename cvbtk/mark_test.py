from dolfin import *
import math as math
import numpy as np

# filepath = "/home/maaike/model/cvbtk/data/mesh_leftventricle_30.hdf5"
filepath ="/home/maaike/Pleunie/sepran_tetra_82422_all_P1.h5"
h5file = HDF5File( mpi_comm_world() , filepath, 'r')
mesh = Mesh(mpi_comm_world())
h5file.read(mesh , 'mesh', True )
V = FunctionSpace(mesh , 'Lagrange',1)

# mesh = UnitCubeMesh(20,20,20)

class boundary(SubDomain):
    """
    Helper method to mark the basal, inner and outer
    surface boundaries.
    """
    def __init__(self):
        self.parameters = self.default_parameters()
        self.point_prm = self.parameters['activation_points']
        self.tol = 1.e-3
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        prm = self.parameters
        point_prm = self.point_prm

        # C = Constant(prm['C'])
        # sig = (1/(2*C))*(sqrt(x[0]**2 + x[1]**2 + (x[2] + C)**2)
        #         + sqrt(x[0]**2 + x[1]**2 + (x[2] - C)**2))

                # tau = 0.5/C*(np.sqrt(x[0]**2 + x[1]**2 + (x[2] + C)**2)
        #         - np.sqrt(x[0]**2 + x[1]**2 + (x[2] - C)**2))

        # ksi = float(math.acosh(sig))
        # phi = atan(x[1]/x[0])
        # theta = float(acos(tau))
        xr, yr, zr = self.ellipsToCartesian(prm['epi_ksi'])
        print(xr,yr,zr)

        r0 = sqrt((x[0]-point_prm['x0'])**2 + (x[1]-point_prm['y0'])**2 + (x[2]-point_prm['z0'])**2)
        r1 = sqrt((x[0]-point_prm['x1'])**2 + (x[1]-point_prm['y1'])**2 + (x[2]-point_prm['z1'])**2)
        r2 = sqrt((x[0]-point_prm['x2'])**2 + (x[1]-point_prm['y2'])**2 + (x[2]-point_prm['z2'])**2)
        r3 = sqrt((x[0]-point_prm['x3'])**2 + (x[1]-point_prm['y3'])**2 + (x[2]-point_prm['z3'])**2)
        r3 = sqrt((x[0]-xr)**2 + (x[1]-yr)**2 + (x[2]-zr)**2)

        # on_endo = abs(ksi - prm['endo_ksi']) <= self.tol
        # on_epi = abs(ksi - prm['epi_ksi']) <= self.tol

        r_act = point_prm['r_act']

        within_r0 = r0 <= r_act
        within_r1 = r1 <= r_act
        within_r2 = r2 <= r_act
        within_r3 = r3 <= r_act

        return on_boundary and ( within_r0 or within_r1 or within_r2 ) or (within_r3 )

        # return ( on_endo and ( within_r0 or within_r1 or within_r2 )) or ( on_epi and within_r3 )

    def ellipsToCartesian(self, ksi,theta=3/5*math.pi,phi = 1/2*math.pi):
        focus = self.parameters['C']
        e = ksi

        # b = focus * sqrt(1-e**2)/e
        a = focus/e
        b = sqrt((1-ksi**2) * a**2)
        eps = math.asinh(b/focus)

        x = focus * sinh(eps) * sin(theta) * cos(phi)
        y = focus * sinh(eps) * sin(theta) * sin(phi)
        z = focus * cosh(eps) * cos(theta)
        return x, y, z

    @staticmethod
    def default_parameters():
        prm = Parameters('left_ventricle')
        prm.add('C', 43)
        prm.add('endo_ksi', 0.375053614389)
        prm.add('epi_ksi', 0.685208)

        point_prm = Parameters('activation_points')

        point_prm.add("x0", 11.1)
        point_prm.add("y0", -11.1)
        point_prm.add("z0", -14)
        point_prm.add("x1", 0.)
        point_prm.add("y1", 16.3)
        point_prm.add("z1", 0)
        point_prm.add("x2", -16.3)
        point_prm.add("y2", 0.)
        point_prm.add("z2", 0)
        point_prm.add("x3", -28.4)
        point_prm.add("y3", 0)
        point_prm.add("z3", -24)
        point_prm.add('r_act', 4)

        prm.add(point_prm)

        return prm

# class points(SubDomain):
#     def inside(self,x,on_boundary):
#         a = Constant(43)
#         sig = (1/(2*a))*( sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+a)*(x[2]+a))+ sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-a)*(x[2]-a)))
#         tau = (1/(2*a))*( sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+a)*(x[2]+a))- sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-a)*(x[2]-a)))
#         ksi = float(math.acosh(sig))
#         phi = atan(x[1]/x[0])
#         theta = float(acos(tau))
#         x0 = 11.1
#         y0 = -11.1
#         z0 = -14
#         x1 = 0.
#         y1 = 16.3
#         z1 = 0
#         x2 = -16.3
#         y2 = 0.
#         z2 = 0
#         x3 = -28.4
#         y3 = 0
#         z3 = -24
#         x4 = 21.5
#         y4 = 0
#         z4 = 25
#         r0 = sqrt((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0)+(x[2]-z0)*(x[2]-z0))
#         r1 = sqrt((x[0]-x1)*(x[0]-x1)+(x[1]-y1)*(x[1]-y1)+(x[2]-z1)*(x[2]-z1))
#         r2 = sqrt((x[0]-x2)*(x[0]-x2)+(x[1]-y2)*(x[1]-y2)+(x[2]-z2)*(x[2]-z2))
#         r3 = sqrt((x[0]-x3)*(x[0]-x3)+(x[1]-y3)*(x[1]-y3)+(x[2]-z3)*(x[2]-z3))
#         r4 = sqrt((x[0]-x4)*(x[0]-x4)+(x[1]-y4)*(x[1]-y4)+(x[2]-z4)*(x[2]-z4))
#         ksi_endo = 0.375053614389
#         on_endo = abs(ksi - ksi_endo ) <= 1.e-2
#         ksi_epi = 0.685208
#         on_epi = abs(ksi - ksi_epi ) <=1.e-2
#         r_act = 4
#         within_r0 = r0<= r_act
#         within_r1 = r1<= r_act
#         within_r2 = r2<= r_act
#         within_r3 = r3<= r_act
#         within_r4 = r4<= r_act
#         return( on_endo and ( within_r0 or within_r1 or within_r2 )) or ( on_epi and within_r3 )
#         return on_boundary and ( within_r0 or within_r1 or within_r2 ) or within_r3

sub_domains = MeshFunction("size_t", mesh,mesh.topology().dim()-1)
sub_domains.set_all(0)

boundary = boundary()
boundary.mark(sub_domains,1)

file = File("subdomains.pvd")
file << sub_domains

td0BC = Constant(0.0)
bc = DirichletBC(V, td0BC , sub_domains, 1, method ='pointwise')




    # Initialize (create) the volume mesh and boundary (surface) mesh.
    # # mesh = self.mesh()
    # b_mesh = BoundaryMesh(mesh, 'exterior')

    # # Initialize the boundary mesh and create arrays of boundary faces.
    # b_map = b_mesh.entity_map(2)
    # b_faces = [Facet(mesh, b_map[cell.index()]) for cell in cells(b_mesh)]

    # # Create an empty list of marker values to fill in.
    # b_value = np.zeros(len(b_faces), dtype=int)

    # # Compute the midpoint locations of each facet on the boundary.
    # X = np.array([each.midpoint() for each in b_faces])

    # # Compute the sigma value of each facet using the midpoint coordinates.
    # C = 4.3
    # sig = 0.5/C*(np.sqrt(X[:, 0]**2 + X[:, 1]**2 + (X[:, 2] + C)**2)
    #                 + np.sqrt(X[:, 0]**2 + X[:, 1]**2 + (X[:, 2] - C)**2))
    # return sig.mean()

    # # Fill in the marker values using sigma in relation to the mean values.
    # # noinspection PyUnresolvedReferences
    # b_value[sig > sig.mean()] = 1 #epicard

    # # noinspection PyUnresolvedReferences
    # b_value[sig < sig.mean()] = 2 #endo

    # # The base is where the z-normal is vertical. Must be called last.
    # n_z = np.array([each.normal()[2] for each in b_faces])
    # b_value[n_z == 1] = 3 #base

    # sub_domains = MeshFunction("size_t", mesh)
    # sub_domains.set_all(0)


    # def base(self, n_z):
    #     return n_z == 1
# class epicard(SubDomain):
#     def inside(self,x,on_boundary):
#         return on_boundary and x[2] <2.4
# sub_domains = MeshFunction("size_t", mesh,mesh.topology().dim()-1)
# sub_domains.set_all(0)

# epi = epicard()
# def endocard(sig):
#     return sig < sig.mean()
# epi = epicard(sig)
# endo = endocard(sig)
# base = base(n_z)

# epi.mark(sub_domains,1)

# endo.mark(sub_domains,2)


    # base.mark(sub_domains,3

    # # Create a FacetFunction and initialize to zero.
    # marker = FacetFunction('size_t', mesh, value=0)

    # # Fill in the FacetFunction with the marked values and return.
    # for f, v in zip(b_faces, b_value):
    #     marker[f] = v
    # return marker
    # return sub_domains

# mark = boundary
# sub_domains = boundary()
# file = File("subdomains.pvd")
# file << sub_domains
# def boundary_epi(x, on_boundary):
#     # Compute the sigma value of each facet using the midpoint coordinates.
#     return on_boundary and marker[x.index] == 1



# class K(Expression):
#     def __init__(self, materials, per, endo, **kwargs):
#         self.materials = materials
#         self.per = per
#         self.endo = endo

#     def eval_cell(self, values, x, cell):
#             if self.materials[cell.index] == 1:
#                 values[0] = self.per
#             elif self.materials[cell.index] == 2:
#                 values[0] = self.endo

# def boundary(x, on_boundary):
#     return on_boundary


# def load_fiber_field(filepath=None, openfile=None, vector_number=0):
#     """
#     Loads the saved fiber vectors ef to the geometry.

#     Args:
#         Either specify the path to the file containing te fiber_vector (filepath) or pass an open file (openfile).
#         vector_number (optional): Specify the vector number to load (if multiple fiber_vector datasets are stored).
#     """
#     if openfile is None:
#         openfile = HDF5File(mpi_comm_world(), filepath, 'r')

#     if openfile.has_dataset('fiber_vector'):
#         # Retrieve element signature
#         attr = openfile.attributes('fiber_vector')
#         element = attr['signature']
#         family = element.split(',')[0].split('(')[-1][1:-1]
#         cell = element.split(',')[1].strip()
#         degree = int(element.split(',')[2])
#         quad_scheme = element.split(',')[3].split('=')[1].split(')')[0][1:-1]
#         # Check if the loaded quadrature degree corresponds to te current quadrature degree.
#         # if degree != parameters['form_compiler']['quadrature_degree']:
#         #     warnings.warn(
#         #         ("\nThe quadrature degree of the loaded fiber vectors (= {}) is not the same as \nthe current " +
#         #         "quadrature degree in parameters['form_compiler']['quadrature_degree']").format(degree))
#         # # Create function space.
#         element_V = VectorElement(family=family,
#                                     cell=cell,
#                                     degree=degree,
#                                     quad_scheme=quad_scheme)
#         V = FunctionSpace(mesh, element_V)
#         ef_func = Function(V)
#         try:
#             # Maybe multiple fiber vectors are saved, load the correct vector.
#             ef_vector = '{}/vector_{}'.format('fiber_vector', vector_number)
#             openfile.read(ef_func, ef_vector)
#         except RuntimeError:
#             ef_vector = 'fiber_vector'
#             openfile.read(ef_func, ef_vector)
#         print('Loading fiber vectors from file, dataset {}'.format(ef_vector))
#     else:
#         print('No fiber field saved.')
#         return

#     # Collect fiber vectors in array.
#     ef_array = ef_func.vector().array().reshape((-1,3))
#     # Compute es and en.
#     es_array = np.zeros(ef_array.shape)
#     en_array = np.zeros(ef_array.shape)
#     for i, ef in enumerate(ef_array):
#         # Compute sheet direction (some vector orthogonal to ef, exact direction does not matter).
#         es = np.array([-ef[1], ef[0], 0])
#         # Normalize.
#         es_array[i, :] = es / np.linalg.norm(es)
#         # Compute sheet-normal direction (orthogonal to ef and es).
#         en = np.cross(ef, es)
#         # Normalize.
#         en_array[i, :] = en / np.linalg.norm(en)
#     return (ef_array, es_array, en_array)



# fibers = load_fiber_field(filepath=filepath)
# print(fibers)
# file = File("marker.pvd")
# file << mark