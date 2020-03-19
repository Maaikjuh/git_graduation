from cvbtk import LeftVentricleMesh
geometry_inputs = {
 #   'cavity_volume': 44.,  
    'focus_height': 4.3,  
    'mesh_resolution': 30.,
    'inner_eccentricity': 0.934819,   
    'mesh_segments': 30,  
    'outer_eccentricity': 0.807075,  
    'truncation_height': 2.4}   
#    'wall_volume': 136.,
 #   'load_fiber_field_from_meshfile':False} 

C = 4.3 #self.parameters['focus_height']
h = 2.4 #self.parameters['truncation_height']
e1 = 0.934819 #self.parameters['inner_eccentricity']
e2 = 0.807075 #self.parameters['outer_eccentricity']
resolution = 30 #self.parameters['mesh_resolution']
segments = 30 #self.parameters['mesh_segments']
LeftVentricleMesh(C, h, e1, e2, resolution, segments=segments)

mesh_name = 'lv_maaike_seg{}_res{}'.format(geometry_inputs['mesh_segments'],
                                                           int(mesh_generator_parameters['mesh_resolution']),)
print_once(mesh_name)
