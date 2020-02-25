# -*- coding: utf-8 -*-
from .basis_vectors import (CardiacBasisVectors, CylindricalBasisVectors,
                            CylindricalBasisVectorsBiV, FiberBasisVectors)
from .coordinate_systems import (CoordinateSystem, EllipsoidalCoordinates,
                                 WallBoundedCoordinates)
from .dataset import Dataset
from .fiber_angles import DefaultFiberAngles
from .geometries import (BaseGeometry, LeftVentricleGeometry,
                         LeftVentricleVADGeometry, ThickWalledSphereGeometry)
from .mechanics import (ActiveStressModel, ArtsBovendeerdActiveStress,
                        ArtsKerckhoffsActiveStress, BovendeerdMaterial,
                        ConstitutiveModel, HolzapfelMaterial,
                        MooneyRivlinMaterial, deformation_gradient,
                        fiber_stretch_ratio, green_lagrange_strain,
                        left_cauchy_green_deformation,
                        right_cauchy_green_deformation)
from .meshes import (LeftVentricleMesh, LeftVentricleVADMesh,
                     ThickWalledSphereMesh)
from .models import LeftVentricleModel, MomentumBalance
from .plotting import HemodynamicsPlot
from .solvers import VolumeSolver
from .utils import (build_nullspace, import_h5, safe_project,
                    vector_space_to_scalar_space,
                    vector_space_to_tensor_space)
from .windkessel import HeartMateII, WindkesselModel, get_phase, kPa_to_mmHg

# Added functions/classes by Tim Hermans
from .basis_vectors import BasisVectorsBayer, GeoFunc
from .geometries import (BiventricleGeometry)
from .fiber_angles import ComputeFiberAngles
from .mechanics import KerckhoffsMaterial
from .meshes import (BiventricleMesh, compute_biventricular_parameters)
from .models import (TimeVaryingElastance, VentriclesWrapper, BiventricleModel,
                     FiberReorientation)
from .plotting import HemodynamicsPlotDC, GeneralHemodynamicsPlot, Export
from .routines import (check_heart_type, check_heart_type_from_inputs, create_materials,
                       create_model, create_windkessel_biv, create_windkessel_lv,
                       load_model_state_from_hdf5, preprocess_biv, preprocess_lv,
                       ReloadState, reset_model_state, set_boundary_conditions,
                       set_initial_conditions_biv, set_initial_conditions_lv,
                       simulate, timestep_biv, timestep_lv, write_data_biv,
                       write_data_lv)
from .solvers import VolumeSolverBiV, CustomNewtonSolver
from .utils import (scalar_space_to_vector_space, save_dict_to_csv,
                    read_dict_from_csv, save_to_disk, print_once,
                    info_once, atan2_, reset_values, global_function_average,
                    figure_make_up, quadrature_function_space)
from .windkessel import (GeneralWindkesselModel, LifetecWindkesselModel,
                         get_phase_dc, mmHg_to_kPa)

__all__ = [
    'CardiacBasisVectors',
    'BasisVectorsBayer',
    'CylindricalBasisVectors',
    'CylindricalBasisVectorsBiV',
    'FiberBasisVectors',
    'GeoFunc',

    'CoordinateSystem',
    'EllipsoidalCoordinates',
    'WallBoundedCoordinates',

    'Dataset',

    'DefaultFiberAngles',
    'ComputeFiberAngles',

    'BaseGeometry',
    'LeftVentricleGeometry',
    'BiventricleGeometry',
    'LeftVentricleVADGeometry',
    'ThickWalledSphereGeometry',

    'ActiveStressModel',
    'ArtsBovendeerdActiveStress',
    'ArtsKerckhoffsActiveStress',
    'BovendeerdMaterial',
    'ConstitutiveModel',
    'KerckhoffsMaterial',
    'HolzapfelMaterial',
    'MooneyRivlinMaterial',
    'deformation_gradient',
    'fiber_stretch_ratio',
    'green_lagrange_strain',
    'left_cauchy_green_deformation',
    'right_cauchy_green_deformation',

    'LeftVentricleMesh',
    'LeftVentricleVADMesh',
    'BiventricleMesh',
    'compute_biventricular_parameters',
    'ThickWalledSphereMesh',

    'TimeVaryingElastance',
    'VentriclesWrapper',
    'LeftVentricleModel',
    'BiventricleModel',
    'MomentumBalance',
    'FiberReorientation',

    'HemodynamicsPlot',
    'HemodynamicsPlotDC',
    'GeneralHemodynamicsPlot',
    'Export',

    'check_heart_type',
    'check_heart_type_from_inputs',
    'create_materials',
    'create_model',
    'create_windkessel_biv',
    'create_windkessel_lv',
    'load_model_state_from_hdf5',
    'preprocess_biv',
    'preprocess_lv',
    'ReloadState',
    'reset_model_state',
    'set_boundary_conditions',
    'set_initial_conditions_biv',
    'set_initial_conditions_lv',
    'simulate',
    'timestep_biv',
    'timestep_lv',
    'write_data_biv',
    'write_data_lv',

    'VolumeSolver',
    'VolumeSolverBiV',
    'CustomNewtonSolver',

    'build_nullspace',
    'import_h5',
    'safe_project',
    'vector_space_to_scalar_space',
    'scalar_space_to_vector_space',
    'vector_space_to_tensor_space',
    'save_dict_to_csv',
    'read_dict_from_csv',
    'save_to_disk',
    'print_once',
    'info_once',
    'atan2_',
    'reset_values',
    'global_function_average',
    'figure_make_up',
    'quadrature_function_space',

    'HeartMateII',
    'WindkesselModel',
    'GeneralWindkesselModel',
    'LifetecWindkesselModel',
    'get_phase',
    'get_phase_dc',
    'kPa_to_mmHg',
    'mmHg_to_kPa'
]

__version__ = '0.0.1'
