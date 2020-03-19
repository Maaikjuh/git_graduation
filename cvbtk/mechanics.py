# -*- coding: utf-8 -*-
"""
This module provides classes and functions related to mechanics.
"""
from dolfin import (And, Constant, interpolate, Expression, DOLFIN_PI, Function, conditional, ge, gt, le,
                    lt, sin, tanh, project, VectorElement, parameters, FunctionSpace)
from dolfin.cpp.common import Parameters
from ufl import Identity, as_tensor, det, dot, exp, grad, inv, sqrt

from .utils import safe_project, vector_space_to_scalar_space, quadrature_function_space


__all__ = [
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
]

def compute_coordinate_expression(degree,element,var =""):
    """
    19-03 Maaike
    Expression for ellipsoidal coordinates
    function obtained from Luca Barbarotta

    Args:
        var : string of ellipsoidal coordinate to be calculated
        V: :class:`~dolfin.FunctionSpace` to define the coordinates on.
        focus: Distance from the origin to the shared focus points.

    Returns:
        Expression for ellipsoidal coordinate
    """

    focus = 4.3

    rastr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]+{f})*(x[2]+{f}))".format(f=focus)
    rbstr = "sqrt(x[0]*x[0]+x[1]*x[1]+(x[2]-{f})*(x[2]-{f}))".format(f=focus)

    taustr= "(1./(2.*{f})*({ra}-{rb}))".format(ra=rastr,rb=rbstr,f=focus)
    sigmastr="(1./(2.*{f})*({ra}+{rb}))".format(ra=rastr,rb=rbstr,f=focus)

    expressions_dict = {"phi": "atan2(x[2],x[0])",
                            "xi": "acosh({sigma})".format(sigma=sigmastr),
                            "theta": "acos({tau})".format(tau=taustr)} 
    return Expression(expressions_dict[var],degree=degree,element=element)


class ConstitutiveModel(object):
    """
    Model-independent methods are defined here to facilitate re-use in new
    constitutive models.

    Args:
        u: The displacement unknown.
        fiber_vectors: Basis vectors (ef, es, en) defining the fiber field.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, u, fiber_vectors, **kwargs):
        self._parameters = self.default_parameters()
        self._parameters.update(kwargs)

        try:
            # Discretize the fiber vectors onto quadrature elements.
            mesh = u.ufl_function_space().ufl_domain().ufl_cargo()

            # Create quadrature space
            # (NOTE that the quadrature degree must have been set in
            # parameters['form_compiler']['quadrature_degree'] before calling this.
            # It will raise an error if it was not set).
            V_q = quadrature_function_space(mesh)

            # Define fiber vectors at quadrature points.
            self._fiber_vectors = [_.to_function(V_q) for _ in fiber_vectors]

        except AttributeError:
            # Fiber vectors are UFL-like definitions that do not need to be discretized.
            self._fiber_vectors = fiber_vectors

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        raise NotImplementedError

    @property
    def parameters(self):
        """
        Return user-defined parameters for this geometry object.
        """
        return self._parameters

    @property
    def fiber_vectors(self):
        """
        Return the discretized fiber basis vectors for this model.
        """
        return self._fiber_vectors

    def cauchy_stress(self, u):
        """
        Return the Cauchy stress tensor.

        Args:
            u: The displacement unknown.

        Returns:
            A UFL-like object.
        """
        F = deformation_gradient(u)
        S = self.piola_kirchhoff2(u)
        return F*S*F.T/det(F)

    def piola_kirchhoff1(self, u):
        """
        Return the 1st Piola-Kirchhoff stress tensor.

        Args:
            u: The displacement unknown.

        Returns:
            A UFL-like object.
        """
        F = deformation_gradient(u)
        S = self.piola_kirchhoff2(u)
        return F*S

    def piola_kirchhoff2(self, u):
        """
        Return the 2nd Piola-Kirchhoff stress tensor.

        Args:
            u: The displacement unknown.

        Returns:
            A UFL-like object.
        """
        raise NotImplementedError

    def von_mises_stress(self, u):
        """
        Return the Von Mises stress.

        Args:
            u: The displacement unknown.

        Returns:
            A UFL-like object.
        """
        C = self.cauchy_stress(u)
        return (0.5*((C[0,0] - C[1,1])**2 + (C[1,1] - C[2,2])**2 + (C[2,2] - C[0,0])**2)
                   + 3*((C[0,1])**2 + (C[1,2])**2 + (C[2,0])**2))**0.5



class ActiveStressModel(ConstitutiveModel):
    """
    Extension of :class:`~cvbtk.ConstitutiveModel` to incorporate common
    additional methods for active stress models.

    Args:
        u: The displacement unknown.
        fiber_vectors: Basis vectors (ef, es, en) defining the fiber field.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, u, fiber_vectors, **kwargs):
        super(ActiveStressModel, self).__init__(u, fiber_vectors, **kwargs)

        # Create time increment and activation map variables.
        self._dt = Constant(1.0)
        self._tact = Constant(0.0 - self.parameters['tdep'])

        # Create, at minimum, a sarcomere length variable.
        ef = self.fiber_vectors[0]
        self._ls = self.parameters['ls0']*fiber_stretch_ratio(u, ef)

        # Create an empty function to hold the old sarcomere lengths.
        Q = vector_space_to_scalar_space(u.ufl_function_space())
        self._ls_old = Function(Q, name='ls_old')

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        raise NotImplementedError

    @property
    def activation_time(self):
        """
        Return the time since/before activation as a variable (Constant).
        """
        return self._tact

    @activation_time.setter
    def activation_time(self, value):
        self._tact.assign(float(value) - self.parameters['tdep'])

    @property
    def dt(self):
        """
        Return the time increment to use for computing shortening velocities as
        a variable (Constant).
        """
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt.assign(float(value))

    def active_stress_scalar(self, u):
        """
        Return the scalar value used to compute the additional active stress
        tensor.

        Args:
            u: The displacement unknown.

        Returns:
            A UFL-like object.
        """
        raise NotImplementedError

    def piola_kirchhoff2(self, u):
        """
        Return the 2nd Piola-Kirchhoff stress tensor.

        Args:
            u: The displacement unknown.

        Returns:
            A UFL-like object.
        """
        # "Derive" the scalar "2nd Piola-Kirchhoff stress" value.
        p = self.active_stress_scalar(u)
        f = self.ls/self.parameters['ls0']
        s = p/f

        # Assemble s into a tensor in the fiber basis.
        S_ = as_tensor(((s, 0, 0), (0, 0, 0), (0, 0, 0)))

        # Rotate S from the fiber basis to the Cartesian basis and return.
        R = as_tensor(self.fiber_vectors)
        S = R.T*S_*R
        return S

    @property
    def ls(self):
        """
        Return the current sarcomere length variable.
        """
        return self._ls

    @property
    def ls_old(self):
        """
        Return the previous sarcomere length variable.
        """
        return self._ls_old

    @ls_old.setter
    def ls_old(self, ls):
        self.ls_old.assign(project(ls, self.ls_old.ufl_function_space()))#, form_compiler_parameters={'quadrature_degree': 10}))

    def upkeep(self):
        """
        Helper method to assign/update model specific quantities at the start of
        a timestep.
        """
        pass


class ArtsBovendeerdActiveStress(ActiveStressModel):
    """
    This is an active stress model proposed by Peter Bovendeerd as a successor
    to the model used by Roy Kerckhoffs (:class:`~cvbtk.ArtsKerckhoffs`).

    Args:
        u: The displacement unknown.
        fiber_vectors: Basis vectors (ef, es, en) defining the fiber field.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, u, fiber_vectors, **kwargs):
        super(ArtsBovendeerdActiveStress, self).__init__(u, fiber_vectors,
                                                         **kwargs)

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters('active_stress')

        prm.add('Ta0', float())

        prm.add('ad', float())
        prm.add('ar', float())
        prm.add('ca', float())
        prm.add('cv', float())

        prm.add('ls0', float())
        prm.add('lsa0', float())
        prm.add('lsa1', float())

        prm.add('taur1', float())
        prm.add('taud1', float())

        prm.add('beta', 0.0)

        prm.add('v0', float())
        prm.add('tdep', float())

        return prm

    def active_stress_scalar(self, u):
        """
        Return the scalar value used to compute the additional active stress
        tensor.

        Args:
            u: The displacement unknown.

        Returns:
            A UFL-like object.
        """
        prm = self.parameters

        # Term for the length dependence.
        iso_term = prm['Ta0']*(tanh(prm['ca']*(self.ls - prm['lsa0'])))**2
        iso_cond = conditional(gt(self.ls, prm['lsa0']), 1, 0)
        f_iso = iso_cond*iso_term

        # Activation time and rise/decay time constants.
        ta = self.activation_time
        tr = prm['taur1'] + prm['ar']*(self.ls - prm['lsa1'])
        td = prm['taud1'] + prm['ad']*(self.ls - prm['lsa1'])

        # Term for the time dependence.
        time_term_1 = (sin(0.5*DOLFIN_PI*ta/tr))**2
        time_term_2 = 1 - (sin(0.5*DOLFIN_PI*(ta - tr)/td))**2
        time_cond_1 = conditional(And(ge(ta, 0), le(ta, tr)), 1, 0)
        time_cond_2 = conditional(And(gt(ta, tr), le(ta, tr + td)), 1, 0)
        f_time = time_cond_1*time_term_1 + time_cond_2*time_term_2

        # Term for the sarcomere shortening velocity dependence.
        vsv0 = (self.ls_old - self.ls)/(self.dt*prm['v0'])
        f_velocity = (1 - vsv0)/(1 + prm['cv']*vsv0)

        # Assemble into the scalar value and return.
        p = f_iso*f_time*f_velocity
        return p

    def upkeep(self):
        """
        Update ls_old with values of ls from the previous timestep.
        """
        self.ls_old = self.ls


class ArtsKerckhoffsActiveStress(ActiveStressModel):
    """
    This is an active stress model used by Peter Bovendeerd and presented in a
    paper by Roy Kerckhoffs.

    Args:
        u: The displacement unknown.
        fiber_vectors: Basis vectors (ef, es, en) defining the fiber field.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """

    def __init__(self, u, fiber_vectors, **kwargs):
        super(ArtsKerckhoffsActiveStress, self).__init__(u, fiber_vectors, **kwargs)
        prm = self.parameters

        # This model requires lc (and lc_old) to be defined.
        Q = vector_space_to_scalar_space(u.ufl_function_space())
        lc_old = Function(Q)
        self._lc_old = lc_old
        lc = lc_old + self.dt*(prm['Ea']*(self.ls_old - lc_old) - 1)*prm['v0']
        
        # 17-03, variable T0 to express the level of active stress generation
        # 18-03 initialize T0
       
        self.T0 = Function(Q)
        self.T0.assign(Constant(self.parameters['Ta0']))


        phi = compute_coordinate_expression(3, Q.ufl_element(),'phi')
        theta = compute_coordinate_expression(3, Q.ufl_element(),'theta')
        xi = compute_coordinate_expression(3, Q.ufl_element(),'xi')

        cpp_exp_Ta0 = "( phi <= {phimax} && phi>= {phimin} && fabs(theta) < {thetar} && xi >= {ximin} )? 0. : {Ta0}".format(Ta0=250., phimin=0., phimax = 0.175, thetar=0.175, ximin=0.5) 
        T0expression = Expression(cpp_exp_Ta0, element=Q.ufl_element(), phi=phi, theta=theta, xi=xi)
        self.T0.interpolate(T0expression)
 

        if prm['restrict_lc']:
            # Upper bound: restrict lc to not be greater than ls.
            max_lc = self.ls  # + 2/prm['Ea']
            lc_ub = conditional(gt(lc, max_lc), 1, 0)
            lc_ub2 = conditional(le(lc, max_lc), 1, 0)
            lc = lc_ub*max_lc + lc_ub2*lc

            # # Lower bound: restrict lc from being lower than 1.5.
            # min_lc = 1.5
            # lc_lb =  conditional(lt(lc, min_lc), 1, 0)
            # lc_lb2 = conditional(ge(lc, min_lc), 1, 0)
            # lc = lc_lb*min_lc + lc_lb2*lc

        # We need logic to decide when lc = ls.
        lc_cond = conditional(ge(self.activation_time, 0), 1, 0)
        ls_cond = conditional(lt(self.activation_time, 0), 1, 0)

        # Store lc and lc_old.
        self._lc = lc_cond*lc + ls_cond*self.ls
        self._lc_old = lc_old

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters('active_stress')

        prm.add('Ta0', float())
        prm.add('Ea', float())
        prm.add('al', float())

        prm.add('lc0', float())
        prm.add('ls0', float())

        prm.add('taur', float())
        prm.add('taud', float())

        prm.add('b', float())
        prm.add('ld', float())

        prm.add('beta', 0.0)

        prm.add('v0', float())
        prm.add('tdep', float())

        prm.add('restrict_lc', False)

        return prm

    def active_stress_scalar(self, u):
        """
        Return the scalar value used to compute the additional active stress
        tensor.

        Args:
            u: The displacement unknown.

        Returns:
            A UFL-like object.
        """
        prm = self.parameters
        




        # Term for the length dependence.
        #iso_term = prm['T0']*(tanh(prm['al']*(self.lc - prm['lc0'])))**2
        iso_cond = conditional(ge(self.lc, prm['lc0']), 1, 0)
        #f_iso = iso_cond*iso_term

        # 17-03, substitute T0 with self.T0
        iso_term = self.T0*(tanh(prm['al']*(self.lc - prm['lc0'])))**2
        f_iso = iso_cond*iso_term

        # Maximum activation time.
        t_max = prm['b']*(self.ls - prm['ld'])

        # Term for the time dependence.
        twitch_term_1 = (tanh(self.activation_time/prm['taur']))**2
        twitch_term_2 = (tanh((t_max - self.activation_time)/prm['taud']))**2
        twitch_cond_1 = ge(self.activation_time, 0)
        twitch_cond_2 = le(self.activation_time, t_max)
        twitch_cond = conditional(And(twitch_cond_1, twitch_cond_2), 1, 0)
        f_twitch = twitch_cond*twitch_term_1*twitch_term_2

        # Assemble into the scalar value and return.
        p = f_iso*f_twitch*prm['Ea']*(self.ls - self.lc)
        return p

    @property
    def lc(self):
        """
        Return the current contractile element length variable.
        """
        return self._lc

    @property
    def lc_old(self):
        """
        Return the previous contractile element length variable.
        """
        return self._lc_old

    @lc_old.setter
    def lc_old(self, lc):
        self.lc_old.assign(safe_project(lc, self.lc_old.ufl_function_space()))

    def upkeep(self):
        """
        Update lc_old and ls_old with values of lc and ls, in that order, from
        the previous timestep.
        """
        self.lc_old = self.lc
        self.ls_old = self.ls


class BovendeerdMaterial(ConstitutiveModel):
    """
    This is a transversely isotropic material model developed by Peter
    Bovendeerd for his work in heart modeling.

    Args:
        u: The displacement unknown.
        fiber_vectors: Basis vectors (ef, es, en) defining the fiber field.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, u, fiber_vectors, **kwargs):
        super(BovendeerdMaterial, self).__init__(u, fiber_vectors, **kwargs)

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters('material_model')

        prm.add('a0', float())
        prm.add('a1', float())
        prm.add('a2', float())
        prm.add('a3', float())
        prm.add('a4', float())
        prm.add('a5', float())

        return prm

    def piola_kirchhoff2(self, u):
        """
        Return the 2nd Piola-Kirchhoff stress tensor.

        Args:
            u: The displacement unknown.

        Returns:
            A UFL-like object.
        """
        a0 = self.parameters['a0']
        a1 = self.parameters['a1']
        a2 = self.parameters['a2']
        a3 = self.parameters['a3']
        a4 = self.parameters['a4']
        a5 = self.parameters['a5']

        # xyz -> fsn
        R = as_tensor(self.fiber_vectors)

        # Green-Lagrange strain in xyz and fsn basis.
        E = green_lagrange_strain(u)
        E_ = R*E*R.T

        # Extract the components of E_fsn.
        E_ff = E_[0, 0]
        E_ss = E_[1, 1]
        E_fs = E_[0, 1]
        E_sf = E_[1, 0]

        # Exponential shape term.
        Q = (a1 + a3)*E_ff**2 + a1*E_ss**2 + 0.5*(a2 + a4)*(E_fs**2 + E_sf**2)

        # Derivative components of the exponential shape term.
        dQ_ff = 2*(a1 + a3)*E_ff
        dQ_ss = 2*a1*E_ss
        dQ_fs = (a2 + a4)*E_fs
        dQ_sf = (a2 + a4)*E_sf

        # Assemble the derivatives into a tensor.
        if u.geometric_dimension() == 2:
            dQ = as_tensor(((dQ_ff, dQ_fs),
                            (dQ_sf, dQ_ss)))

        else:
            E_nn = E_[2, 2]
            E_fn = E_[0, 2]
            E_nf = E_[2, 0]
            E_sn = E_[1, 2]
            E_ns = E_[2, 1]
            Q += a1*E_nn**2 + 0.5*a2*(E_ns**2 + E_sn**2) + 0.5*(a2 + a4)*(E_fn**2 + E_nf**2)
            dQ_nn = 2*a1*E_nn
            dQ_fn = (a2 + a4)*E_fn
            dQ_nf = (a2 + a4)*E_nf
            dQ_sn = a2*E_sn
            dQ_ns = a2*E_ns
            dQ = as_tensor(((dQ_ff, dQ_fs, dQ_fn),
                            (dQ_sf, dQ_ss, dQ_sn),
                            (dQ_nf, dQ_ns, dQ_nn)))

        # Shape term component of S in fsn basis.
        Ss_ = a0*exp(Q)*dQ

        # Right Cauchy-Green strain in fsn basis.
        # noinspection PyTypeChecker
        C_ = 2*E_ + Identity(u.geometric_dimension())

        if u.geometric_dimension() == 2:
            Wv_ = 2*det(C_)*inv(C_.T)

        else:
            # Extract the components of C_fsn.
            C_ff = C_[0, 0]
            C_ss = C_[1, 1]
            C_fs = C_[0, 1]
            C_sf = C_[1, 0]
            C_nn = C_[2, 2]
            C_fn = C_[0, 2]
            C_nf = C_[2, 0]
            C_sn = C_[1, 2]
            C_ns = C_[2, 1]

            # Volume term components of S in fsn basis.
            Wv_ff = 2*(C_ss*C_nn - C_sn*C_ns)
            Wv_ss = 2*(C_ff*C_nn - C_fn*C_nf)
            Wv_fs = 2*(C_sn*C_nf - C_sf*C_nn)
            Wv_nn = 2*(C_ff*C_ss - C_fs*C_sf)
            Wv_sn = 2*(C_fs*C_nf - C_ff*C_ns)
            Wv_nf = 2*(C_fs*C_sn - C_fn*C_ss)
            Wv_sf = 2*(C_fn*C_ns - C_fs*C_nn)
            Wv_ns = 2*(C_fn*C_sf - C_ff*C_sn)
            Wv_fn = 2*(C_sf*C_ns - C_ss*C_nf)

            # Assemble the components into a tensor.
            Wv_ = as_tensor(((Wv_ff, Wv_fs, Wv_fn),
                             (Wv_sf, Wv_ss, Wv_sn),
                             (Wv_nf, Wv_ns, Wv_nn)))

        # Volume term component of S in fsn basis.
        Sv_ = 2*a5*(det(C_) - 1)*Wv_

        # Combine the terms, rotate to Cartesian basis, and return.
        S = R.T*(Ss_ + Sv_)*R

        return S

class KerckhoffsMaterial(ConstitutiveModel):
    """
    Ppassive material law according to Kerckhoffs (2003).

    Args:
        u: The displacement unknown.
        fiber_vectors: Basis vectors (ef, es, en) defining the fiber field.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, u, fiber_vectors, **kwargs):
        super(KerckhoffsMaterial, self).__init__(u, fiber_vectors, **kwargs)

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters('material_model')

        prm.add('a0', float())
        prm.add('a1', float())
        prm.add('a2', float())
        prm.add('a3', float())
        prm.add('a4', float())
        prm.add('a5', float())

        return prm

    def piola_kirchhoff2(self, u):
        """
        Return the 2nd Piola-Kirchhoff stress tensor.

        Args:
            u: The displacement unknown.

        Returns:
            A UFL-like object.
        """
        a0 = self.parameters['a0']
        a1 = self.parameters['a1']
        a2 = self.parameters['a2']
        a3 = self.parameters['a3']
        a4 = self.parameters['a4']
        a5 = self.parameters['a5']

        # xyz -> fsn
        R = as_tensor(self.fiber_vectors)

        # Green-Lagrange strain in xyz and fsn basis.
        E = green_lagrange_strain(u)
        E_ = R*E*R.T

        # Extract the components of E_fsn.
        E_ff = E_[0, 0]
        E_ss = E_[1, 1]
        E_fs = E_[0, 1]
        E_sf = E_[1, 0]
        E_nn = E_[2, 2]
        E_fn = E_[0, 2]
        E_nf = E_[2, 0]
        E_sn = E_[1, 2]
        E_ns = E_[2, 1]

        # Exponential shape term isotropic part.
        Qi = a1*(E_ff*E_ff + E_ss*E_ss + E_nn*E_nn) \
           + a2*(E_fs*E_sf + E_fn*E_nf + E_sn*E_ns) \
           + (2*a1 - a2)*(E_ff*E_ss + E_ff*E_nn + E_ss*E_nn)

        # Derivative components of the exponential shape term isotropic part.
        dQi_ff = 2*a1*(E_ff + E_ss + E_nn) - a2*(E_ss + E_nn)
        dQi_fs = a2*E_sf
        dQi_fn = a2*E_nf
        dQi_sf = a2*E_fs
        dQi_ss = 2*a1*(E_ss + E_ff + E_nn) - a2*(E_ff + E_nn)
        dQi_sn = a2*E_ns
        dQi_nf = a2*E_fn
        dQi_ns = a2*E_sn
        dQi_nn = 2*a1*(E_nn + E_ff + E_ss) - a2*(E_ff + E_ss)

        # Assemble the derivatives into a tensor.
        dQi = as_tensor(((dQi_ff, dQi_fs, dQi_fn),
                        (dQi_sf, dQi_ss, dQi_sn),
                        (dQi_nf, dQi_ns, dQi_nn)))

        # Exponential shape term fiber part.
        Qf = a4*E_ff*E_ff

        # Derivative components of the exponential shape term fiber part.
        dQf_ff = 2*a4*E_ff

        # Assemble the derivatives into a tensor.
        dQf = as_tensor(((dQf_ff, 0, 0),
                        (0, 0, 0),
                        (0, 0, 0)))

        # Total shape term component of S in fsn basis.
        Ss_ = a0*exp(Qi)*dQi + a3*exp(Qf)*dQf

        # Right Cauchy-Green strain in fsn basis.
        # noinspection PyTypeChecker
        C_ = 2*E_ + Identity(u.geometric_dimension())

        if u.geometric_dimension() == 2:
            Wv_ = 2*det(C_)*inv(C_.T)

        else:
            # Extract the components of C_fsn.
            C_ff = C_[0, 0]
            C_ss = C_[1, 1]
            C_fs = C_[0, 1]
            C_sf = C_[1, 0]
            C_nn = C_[2, 2]
            C_fn = C_[0, 2]
            C_nf = C_[2, 0]
            C_sn = C_[1, 2]
            C_ns = C_[2, 1]

            # Volume term components of S in fsn basis.
            Wv_ff = 2*(C_ss*C_nn - C_sn*C_ns)
            Wv_ss = 2*(C_ff*C_nn - C_fn*C_nf)
            Wv_fs = 2*(C_sn*C_nf - C_sf*C_nn)
            Wv_nn = 2*(C_ff*C_ss - C_fs*C_sf)
            Wv_sn = 2*(C_fs*C_nf - C_ff*C_ns)
            Wv_nf = 2*(C_fs*C_sn - C_fn*C_ss)
            Wv_sf = 2*(C_fn*C_ns - C_fs*C_nn)
            Wv_ns = 2*(C_fn*C_sf - C_ff*C_sn)
            Wv_fn = 2*(C_sf*C_ns - C_ss*C_nf)

            # Assemble the components into a tensor.
            Wv_ = as_tensor(((Wv_ff, Wv_fs, Wv_fn),
                             (Wv_sf, Wv_ss, Wv_sn),
                             (Wv_nf, Wv_ns, Wv_nn)))

        # Volume term component of S in fsn basis.
        Sv_ = 2*a5*(det(C_) - 1)*Wv_

        # Combine the terms, rotate to Cartesian basis, and return.
        S = R.T*(Ss_ + Sv_)*R

        return S


class HolzapfelMaterial(ConstitutiveModel):
    """
    This is an orthotropic material model developed by Ogden and Holzapfel.

    Args:
        u: The displacement unknown.
        fiber_vectors: Basis vectors (ef, es, en) defining the fiber field.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, u, fiber_vectors, **kwargs):
        super(HolzapfelMaterial, self).__init__(u, fiber_vectors, **kwargs)

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters('material_model')

        prm.add('a0', float())
        prm.add('a1', float())
        prm.add('a2', float())
        prm.add('a3', float())

        prm.add('b0', 1.0)
        prm.add('b1', 1.0)
        prm.add('b2', 1.0)
        prm.add('b3', 1.0)

        return prm

    def piola_kirchhoff2(self, u):
        """
        Return the 2nd Piola-Kirchhoff stress tensor.

        Args:
            u: The displacement unknown.

        Returns:
            A UFL-like object.
        """
        a0, b0 = self.parameters['a0'], self.parameters['b0']
        a1, b1 = self.parameters['a1'], self.parameters['b1']
        a2, b2 = self.parameters['a2'], self.parameters['b2']
        a3, b3 = self.parameters['a3'], self.parameters['b3']

        # xyz -> fsn
        R = as_tensor(self.fiber_vectors)

        # Right Cauchy-Green deformation in xyz and fsn basis.
        C = right_cauchy_green_deformation(u)
        C_ = R*C*R.T

        # Invariants of B for the shape term.
        I4_ff = C_[0, 0]
        I4_ss = C_[1, 1]
        I4_nn = C_[2, 2]
        I8_fs = C_[0, 1]
        I1 = I4_ff + I4_ss + I4_nn

        # Derivatives of the invariants of C.
        # TODO Double check this formulation.
        ef, es, _ = self.fiber_vectors
        dW_I1 = a0*exp(b0*(I1 - 3))*Identity(u.geometric_dimension())
        dW_I4_ff = 2*a1*exp(b1*(I4_ff - 1)**2)*(I4_ff - 1)
        dW_I4_ss = 2*a2*exp(b2*(I4_ss - 1)**2)*(I4_ss - 1)
        dW_I8_fs = a3*exp(b3*I8_fs**2)*I8_fs*dot(ef, es)

        # Shape term component of S in fsn basis.
        Ss_ = 2*as_tensor(((dW_I1 + dW_I4_ff, dW_I8_fs, 0),
                           (dW_I8_fs, dW_I1 + dW_I4_ss, 0),
                           (0, 0, dW_I1)))

        # Combine the terms, rotate to Cartesian basis, and return.
        S = R.T*Ss_*R
        return S


class MooneyRivlinMaterial(ConstitutiveModel):
    """
    This is an isotropic material model developed by Mooney and Rivlin.

    Args:
        u: The displacement unknown.
        fiber_vectors: Basis vectors (ef, es, en) defining the fiber field.
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, u, fiber_vectors, **kwargs):
        super(MooneyRivlinMaterial, self).__init__(u, fiber_vectors, **kwargs)

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters('material_model')

        prm.add('a0', float())
        prm.add('a1', float())
        prm.add('a2', float())

        return prm

    def piola_kirchhoff2(self, u):
        """
        Return the 2nd Piola-Kirchhoff stress tensor.

        Args:
            u: The displacement unknown.

        Returns:
            A UFL-like object.
        """
        a0 = self.parameters['a0']
        a1 = self.parameters['a1']
        a2 = self.parameters['a2']

        # xyz -> fsn
        R = as_tensor(self.fiber_vectors)

        # Left Cauchy-Green deformation in xyz and fsn basis.
        B = left_cauchy_green_deformation(u)
        B_ = R*B*R.T

        # Extract the components of B_fsn.
        B_ff = B_[0, 0]
        B_ss = B_[1, 1]
        B_nn = B_[2, 2]

        # Invariants of B for the shape term.
        I1 = B_ff + B_ss + B_nn

        # Derivatives of the invariants of B.
        dI1 = a0*Identity(u.geometric_dimension())
        dI2 = a1*(I1*Identity(u.geometric_dimension()) - B_)  # B.T = B

        # Shape term component of S in fsn basis.
        Ss_ = 2*(dI1 + dI2)

        # Deformation gradient in xyz and fsn basis.
        F = deformation_gradient(u)
        F_ = R*F*R.T

        # Volume term component of S in fsn basis.
        I3 = det(F_)
        dWv = 2*(a2*(I3 - 1)*I3*inv(F_.T) - (a0 + 2*a1)/I3)
        Sv_ = 2*dWv

        # Combine the terms, rotate to Cartesian basis, and return.
        S = R.T*(Ss_ + Sv_)*R
        return S


def deformation_gradient(u):
    """
    Return the deformation gradient tensor.

    Args:
        u: The displacement unknown.

    Returns:
        A UFL-like object.
    """
    return grad(u) + Identity(u.geometric_dimension())


def fiber_stretch_ratio(u, ef):
    """
    Return the stretch ratio in the fiber direction.

    Args:
        u: The displacement unknown.
        ef: Fiber-aligned basis vector.

    Returns:
        A UFL-like object.
    """
    C = right_cauchy_green_deformation(u)
    return sqrt(dot(ef, C*ef))


def green_lagrange_strain(u):
    """
    Return the Green-Lagrange strain tensor.

    Args:
        u: The displacement unknown.

    Returns:
        A UFL-like object.
    """
    F = deformation_gradient(u)
    return 0.5*(F.T*F - Identity(u.geometric_dimension()))


def left_cauchy_green_deformation(u):
    """
    Return the left Cauchy-Green deformation tensor.

    Args:
        u: The displacement unknown.

    Returns:
        A UFL-like object.
    """
    F = deformation_gradient(u)
    return F*F.T


def right_cauchy_green_deformation(u):
    """
    Return the right Cauchy-Green deformation tensor.

    Args:
        u: The displacement unknown.

    Returns:
        A UFL-like object.
    """
    F = deformation_gradient(u)
    return F.T*F
