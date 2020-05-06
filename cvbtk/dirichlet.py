# -*- coding: utf-8 -*-
"""TODO DirichletBC stuff"""
import fenics

__all__ = ['DirichletBC', 'get_dirichlet_bc']


class DirichletBC(fenics.DirichletBC):

    def __init__(self,*args,**kwargs): #V, g, sub_domains, sub_domain,method='topological'):
        g = args[1] if type(args[1])==fenics.GenericFunction else None
        self.g = g  # save 'g' since there isn't a way to access it later
        super(DirichletBC, self).__init__(*args, **kwargs)

    def update(self, c=None, k=None, t=None):
        """Updates :class:`beatit.DirichletBC` :attribute:`g` expression.

        Args:
            c (float, optional): Constant scalar value.
            k (float, optional): Variable scalar value.
            t (int, optional): Simulation time as int, converted to float.
        """
        # Update constant scalar value.
        if c:
            self._update_c(c)
        # Update variable scalar value.
        if k:
            self._update_k(k)
        # Update simulation time value.
        if t:
            self._update_t(t)

    def _update_c(self, c):
        """Updates :class:`beatit.DirichletBC` :attribute:`g` with new ``c``.

        Args:
            c (float): Constant scalar value.
        """
        if isinstance(self.value(), fenics.GenericFunction):
            self.set_value(fenics.Constant(c))

    def _update_k(self, k):
        """Updates :class:`beatit.DirichletBC` :attribute:`g` with new ``k``.

        Args:
            k (float): Variable scalar value.
        """
        try:
            self.g.k = k
        except AttributeError:
            pass

    def _update_t(self, t):
        """Updates :class:`beatit.DirichletBC` :attribute:`g` with new ``t``.

        Args:
            t (int): Simulation time as int, converted to float.
        """
        try:
            self.g.t = t
        except AttributeError:
            pass


def get_dirichlet_bc(V, inputs, sub_domains=None):
    """Returns a :class:`fenics.DirichletBC` object."""
    if isinstance(inputs, dict):
        try:
            inputs['g']
            return _get_dirichlet_bc_solo(V, sub_domains=sub_domains, **inputs)
        except KeyError:
            list_input = []
            for key, value in inputs.items():
                list_input.append(value)
            return _get_dirichlet_bc_many(V, sub_domains=sub_domains, inputs=list_input)
    elif isinstance(inputs, list):
        return _get_dirichlet_bc_many(V, sub_domains=sub_domains, inputs=inputs)


def _get_dirichlet_bc_many(V, inputs, sub_domains=None):
    """Returns a list of :class:`fenics.DirichletBC` objects."""
    bcs = []
    for item in inputs:
        bcs.append(_get_dirichlet_bc_solo(V, sub_domains=sub_domains, **item))
    return bcs


def _get_dirichlet_bc_solo(V, *args, **kwargs ):
    """Returns a :class:`fenics.DirichletBC` object."""
    # Get the proper subspace of V.
    g = kwargs.pop('g')
    component = kwargs.pop('component')
    sub_domain = kwargs['sub_domain']
    sub_domains = kwargs['sub_domains']
    method = kwargs['method'] if 'method' in kwargs else 'topological'
    check_midpoint = kwargs['check_midpoint'] if 'check_midpoint' in kwargs else False
    if V.num_sub_spaces() > 0:
        V_ = V.sub(component)
    else:
        V_ = V
    g_ = g
    # The most simple DirichletBC is one where g is a constant/scalar value.
    try:
        g_ = float(g)
    # Assume an expression is wanted if a ValueError is thrown.
    except ValueError:
        degree = V_.ufl_element().degree() + 1
        g_ = fenics.Expression(g, degree=degree, t=0.0, k=0.0)
    if type(sub_domains)==fenics.SubDomain:
        sub_domains = kwargs.pop('sub_domains')
        bc = DirichletBC(V_, g_, sub_domains, **kwargs)
    elif type(sub_domains)==fenics.MeshFunctionSizet:
        sub_domains = kwargs.pop('sub_domains')
        sub_domain = kwargs.pop('sub_domain')
        bc = DirichletBC(V_, g_, sub_domains, sub_domain, **kwargs)
    elif sub_domains==None:
        sub_domain = kwargs.pop('sub_domain')
        bc = DirichletBC(V_, g_, sub_domain, **kwargs)
    else:
        fenics.log(fenics.INFO,"ERROR: wrong parameters for Dirichlet BC")
        exit
    return bc
