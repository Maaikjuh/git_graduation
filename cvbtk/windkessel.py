# -*- coding: utf-8 -*-
"""
This module provides classes that define windkessel models and related objects.
"""
from dolfin.cpp.common import Parameters

__all__ = [
    'HeartMateII',
    'WindkesselModel',
    'GeneralWindkesselModel',
    'LifetecWindkesselModel',
    'get_phase',
    'get_phase_dc',
    'kPa_to_mmHg',
    'mmHg_to_kPa'
]


class HeartMateII(object):
    """
    This class represents a HeartMate II left ventricular assist device.

    Args:
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """

    def __init__(self, **kwargs):
        self._parameters = self.default_parameters()
        self._parameters.update(kwargs)

        self._q = 0.0
        self._v = self._parameters['lvad_volume']

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters('lvad_model')

        prm.add('lvad_volume', 66.0)

        prm.add('alpha_slope', 0.0091)
        prm.add('alpha_intercept', 1.4)

        prm.add('beta_slope', -0.19)
        prm.add('beta_intercept', -1.9)

        prm.add('frequency', float())

        return prm

    @property
    def parameters(self):
        """
        Return user-defined parameters for this model object.
        """
        return self._parameters

    def compute_flowrate(self, windkessel_pressures, cavity_pressures):
        """
        Compute the flowrate of the pump given a set of windkessel and cavity
        pressures.

        Args:
            windkessel_pressures: Dictionary of windkessel pressures.
            cavity_pressures: Dictionary of cavity pressures.

        Returns:
            Dictionary of computed flowrates.
        """
        # Extract the relevant pressures from the inputs.
        pcav = cavity_pressures['lv']
        part = windkessel_pressures['art']

        # Extract relevant model parameters.
        frequency = self.parameters['frequency']
        a_slope = self.parameters['alpha_slope']
        a_intercept = self.parameters['alpha_intercept']
        b_slope = self.parameters['beta_slope']
        b_intercept = self.parameters['beta_intercept']

        dp = kPa_to_mmHg(part - pcav)
        q = (a_slope*dp + a_intercept)*frequency + b_slope*dp + b_intercept
        return {'lvad': q/60}  # L/min to ml/ms

    @property
    def volume(self):
        """
        Return a dictionary of volumes for the current state.
        """
        return {'lvad': self._v}

    @property
    def flowrate(self):
        """
        Return a dictionary of flowrates for the current state.
        """
        return {'lvad': self._q}

    @flowrate.setter
    def flowrate(self, values):
        """
        Set the flowrates of the current state from a given dictionary of
        flowrates.
        """
        self._q = float(values.get('lvad', self._q))


class WindkesselModel(object):
    """
    This class represents two three-element windkessel models representing the
    arterial and venous system in bulk, along with aortic valve, mitral valve,
    and peripheral flowrates.

    Args:
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, **kwargs):
        self._parameters = self.default_parameters()
        self._parameters.update(kwargs)

        self._qao = 0.0
        self._qmv = 0.0
        self._qper = 0.0
        self._vart = 0.0
        self._vven = 0.0
        self._part = 0.0
        self._pven = 0.0

        self._lvad = None

    @staticmethod
    def default_parameters():
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters('windkessel_model')

        prm.add('total_volume', float())

        prm.add('venous_compliance', float())
        prm.add('arterial_compliance', float())

        prm.add('venous_resistance', float())
        prm.add('arterial_resistance', float())
        prm.add('peripheral_resistance', float())

        prm.add('venous_resting_volume', float())
        prm.add('arterial_resting_volume', float())

        return prm

    @property
    def parameters(self):
        """
        Return user-defined parameters for this model object.
        """
        return self._parameters

    def compute_flowrate(self, windkessel_pressures, cavity_pressures):
        """
        Compute the flowrate of the system given a set of windkessel and cavity
        pressures.

        Args:
            windkessel_pressures: Dictionary of windkessel pressures.
            cavity_pressures: Dictionary of cavity pressures.

        Returns:
            Dictionary of computed flowrates.
        """
        # Extract the relevant pressures from the inputs.
        pcav = cavity_pressures['lv']
        part = windkessel_pressures['art']
        pven = windkessel_pressures['ven']

        # Extract relevant model parameters.
        rven = self.parameters['venous_resistance']
        rart = self.parameters['arterial_resistance']
        rper = self.parameters['peripheral_resistance']

        # Compute the mitral, aortic, and peripheral flowrates.
        q = {'mv': max((pven - pcav)/rven, 0.0),
             'ao': max((pcav - part)/rart, 0.0),
             'per': max((part - pven)/rper, 0.0)}

        # Compute the LVAD flowrate if this model has an LVAD defined.
        if self._lvad is not None:
            q['lvad'] = self._lvad.compute_flowrate(windkessel_pressures,
                                                    cavity_pressures)['lvad']

        return q

    def compute_pressure(self, windkessel_volumes):
        """
        Compute the pressure of the system given a set of windkessel volumes.

        Args:
            windkessel_volumes: Dictionary of windkessel volumes.

        Returns:
            Dictionary of computed pressures.
        """
        # Extract the relevant volumes from the inputs.
        vart = windkessel_volumes['art']
        vven = windkessel_volumes['ven']

        # Extract relevant model parameters.
        cven = self.parameters['venous_compliance']
        cart = self.parameters['arterial_compliance']
        vven_rest = self.parameters['venous_resting_volume']
        vart_rest = self.parameters['arterial_resting_volume']

        # Compute the venous and arterial pressures.
        p = {'art': (vart - vart_rest)/cart, 'ven': (vven - vven_rest)/cven}
        return p

    def compute_volume(self, windkessel_pressures):
        """
        Compute the volume of the system given a set of windkessel pressures.

        Args:
            windkessel_pressures: Dictionary of windkessel pressures.

        Returns:
            Dictionary of computed volumes.
        """
        # Extract relevant model parameters.
        cart = self.parameters['arterial_compliance']
        vart_rest = self.parameters['arterial_resting_volume']
        return {'art': windkessel_pressures['art']*cart + vart_rest}

    @property
    def flowrate(self):
        """
        Return a dictionary of flowrates for the current state.
        """
        q = {'ao': self._qao, 'mv': self._qmv, 'per': self._qper}
        if self._lvad is not None:
            q['lvad'] = self._lvad.flowrate['lvad']
        return q

    @flowrate.setter
    def flowrate(self, values):
        """
        Set the flowrates of the current state from a given dictionary of
        flowrates.
        """
        self._qao = float(values.get('ao', self._qao))
        self._qmv = float(values.get('mv', self._qmv))
        self._qper = float(values.get('per', self._qper))
        if self._lvad is not None:
            q_lvad = values.get('lvad', self._lvad.flowrate['lvad'])
            self._lvad.flowrate = {'lvad': q_lvad}

    @property
    def pressure(self):
        """
        Return a dictionary of pressures for the current state.
        """
        return {'art': self._part, 'ven': self._pven}

    @pressure.setter
    def pressure(self, values):
        """
        Set the pressures of the current state from a given dictionary of
        pressures.
        """
        self._part = float(values.get('art', self._part))
        self._pven = float(values.get('ven', self._pven))

    @property
    def volume(self):
        """
        Return a dictionary of volumes for the current state.
        """
        v = {'art': self._vart, 'ven': self._vven}
        if self._lvad is not None:
            v['lvad'] = self._lvad.volume['lvad']
        return v

    @volume.setter
    def volume(self, values):
        """
        Set the volume of the current state from a given dictionary of volumes.
        """
        self._vart = float(values.get('art', self._vart))
        self._vven = float(values.get('ven', self._vven))

    @property
    def lvad(self):
        """
        Return the attached LVAD object if one exists, else ``None``.
        """
        return self._lvad

    @lvad.setter
    def lvad(self, value):
        """
        Attach an LVAD to this windkessel model.
        """
        self._lvad = value


class GeneralWindkesselModel(object):
    """
    This class represents a three-element windkessel model representing the
    arterial and venous system in bulk, along with arterial, venous,
    and peripheral flowrates.

    Args:
        name (str, optional): Name for the parameter set
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, name = 'windkessel_model', **kwargs):
        self._parameters = self.default_parameters(name)
        self._parameters.update(kwargs)

        self._qart = 0.0
        self._qven = 0.0
        self._qper = 0.0
        self._vart = 0.0
        self._vven = 0.0
        self._part = 0.0
        self._pven = 0.0

        self._lvad = None

    @staticmethod
    def default_parameters(name):
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters(name)

        prm.add('venous_compliance', float())
        prm.add('arterial_compliance', float())

        prm.add('venous_resistance', float())
        prm.add('arterial_resistance', float())
        prm.add('peripheral_resistance', float())

        prm.add('venous_resting_volume', float())
        prm.add('arterial_resting_volume', float())

        return prm

    @property
    def parameters(self):
        """
        Return user-defined parameters for this model object.
        """
        return self._parameters

    def compute_flowrate(self, boundary_pressures, windkessel_pressures = None):
        """
        Compute the flowrate of the system given a set of windkessel and cavity
        pressures.

        Args:
            boundary_pressures (dict): Dictionary of pressure at inflow and outflow boundary of windkessel (with keys 'in' and 'out')
            windkessel_pressures (dict, optional): Dictionary of windkessel pressures. The current state is used by default.
        Returns:
            Dictionary of computed flowrates.
        """
        # Extract the relevant pressures from the inputs.
        pin = boundary_pressures['in']
        pout = boundary_pressures['out']
        if windkessel_pressures is None:
            part = self.pressure['art']
            pven = self.pressure['ven']
        else:
            part = windkessel_pressures['art']
            pven = windkessel_pressures['ven']

        # Extract relevant model parameters.
        rven = self.parameters['venous_resistance']
        rart = self.parameters['arterial_resistance']
        rper = self.parameters['peripheral_resistance']

        # Compute the mitral, aortic, and peripheral flowrates.
        q = {'ven': max((pven - pout)/rven, 0.0),
             'art': max((pin - part)/rart, 0.0),
             'per': (part - pven)/rper}

        # Compute the LVAD flowrate if this model has an LVAD defined.
        if self._lvad is not None:
            # assume that inflow pressure is left ventricular pressure if LVAD is present
            cavity_pressures = {'lv': pin}
            windkessel_pressures = {'art': part,
                                    'ven': pven}
            q['lvad'] = self._lvad.compute_flowrate(windkessel_pressures,
                                                    cavity_pressures)['lvad']

        return q

    def compute_pressure(self, windkessel_volumes = None):
        """
        Compute the pressure of the system given a set of windkessel volumes.

        Args:
            windkessel_volumes (optional): Dictionary of windkessel volumes. The current state is used by default.

        Returns:
            Dictionary of computed pressures.
        """
        # Extract the relevant volumes from the inputs.
        if windkessel_volumes is None:
            vart = self.volume['art']
            vven = self.volume['ven']
        else:
            vart = windkessel_volumes['art']
            vven = windkessel_volumes['ven']

        # Extract relevant model parameters.
        cven = self.parameters['venous_compliance']
        cart = self.parameters['arterial_compliance']
        vven_rest = self.parameters['venous_resting_volume']
        vart_rest = self.parameters['arterial_resting_volume']

        # Compute the venous and arterial pressures.
        p = {'art': (vart - vart_rest)/cart, 'ven': (vven - vven_rest)/cven}
        return p

    def compute_volume(self, windkessel_pressures = None):
        """
        Compute the volume of the system given a set of windkessel pressures.

        Args:
            windkessel_pressures: Dictionary of windkessel pressures. The current state is used by default.

        Returns:
            Dictionary of computed volumes.
        """
        # Extract the relevant volumes from the inputs.
        if windkessel_pressures is None:
            part = self.pressure['art']
            pven = self.pressure['ven']
        else:
            part = windkessel_pressures['art']
            pven = windkessel_pressures['ven']

        # Extract relevant model parameters.
        cven = self.parameters['venous_compliance']
        cart = self.parameters['arterial_compliance']
        vven_rest = self.parameters['venous_resting_volume']
        vart_rest = self.parameters['arterial_resting_volume']

        # Compute the venous and arterial volumes.
        v = {'art': part*cart + vart_rest, 'ven': pven*cven + vven_rest}
        return v

    @property
    def flowrate(self):
        """
        Return a dictionary of flowrates for the current state.
        """
        q = {'art': self._qart, 'ven': self._qven, 'per': self._qper}
        if self._lvad is not None:
            q['lvad'] = self._lvad.flowrate['lvad']
        return q

    @flowrate.setter
    def flowrate(self, values):
        """
        Set the flowrates of the current state from a given dictionary of
        flowrates.
        """
        self._qart = float(values.get('art', self._qart))
        self._qven = float(values.get('ven', self._qven))
        self._qper = float(values.get('per', self._qper))
        if self._lvad is not None:
            q_lvad = values.get('lvad', self._lvad.flowrate['lvad'])
            self._lvad.flowrate = {'lvad': q_lvad}

    @property
    def pressure(self):
        """
        Return a dictionary of pressures for the current state.
        """
        return {'art': self._part, 'ven': self._pven}

    @pressure.setter
    def pressure(self, values):
        """
        Set the pressures of the current state from a given dictionary of
        pressures.
        """
        self._part = float(values.get('art', self._part))
        self._pven = float(values.get('ven', self._pven))

    @property
    def volume(self):
        """
        Return a dictionary of volumes for the current state.
        """
        v = {'art': self._vart, 'ven': self._vven}
        if self._lvad is not None:
            v['lvad'] = self._lvad.volume['lvad']
        return v

    @volume.setter
    def volume(self, values):
        """
        Set the volume of the current state from a given dictionary of volumes.
        """
        self._vart = float(values.get('art', self._vart))
        self._vven = float(values.get('ven', self._vven))

    @property
    def lvad(self):
        """
        Return the attached LVAD object if one exists, else ``None``.
        """
        return self._lvad

    @lvad.setter
    def lvad(self, value):
        """
        Attach an LVAD to this windkessel model.
        """
        self._lvad = value


class LifetecWindkesselModel(object):
    """
    This class represents a constant preload and a three-element windkessel afetrload model representing the
    arterial and venous system in bulk, along with arterial, venous, and peripheral flowrates.

    Args:
        name (str, optional): Name for the parameter set
        **kwargs: Arbitrary keyword arguments for user-defined parameters.
    """
    def __init__(self, name = 'windkessel_model', **kwargs):
        self._parameters = self.default_parameters(name)
        self._parameters.update(kwargs)

        self._qao = 0.0
        self._qmv = 0.0
        self._qper = 0.0
        self._vart = 0.0
        self._vven = 0.0
        self._part = 0.0
        self._pven = self.parameters['venous_pressure']

        self._lvad = None

    @staticmethod
    def default_parameters(name):
        """
        Return a set of default parameters for this model.
        """
        prm = Parameters(name)

        prm.add('total_volume', 5000.0)  # Not important for non-closed loop. Included for compatibility.

        prm.add('venous_pressure', float())

        prm.add('arterial_compliance', float())

        prm.add('venous_resistance', float())
        prm.add('arterial_resistance', float())
        prm.add('peripheral_resistance', float())

        return prm

    @property
    def parameters(self):
        """
        Return user-defined parameters for this model object.
        """
        return self._parameters

    def compute_flowrate(self, windkessel_pressures, cavity_pressures):
        """
        Compute the flowrate of the system given a set of windkessel and cavity
        pressures.

        Args:
            windkessel_pressures: Dictionary of windkessel pressures.
            cavity_pressures: Dictionary of cavity pressures.

        Returns:
            Dictionary of computed flowrates.
        """
        # Extract the relevant pressures from the inputs.
        pcav = cavity_pressures['lv']
        part = windkessel_pressures['art']
        pven = windkessel_pressures['ven']

        # Extract relevant model parameters.
        rven = self.parameters['venous_resistance']
        rart = self.parameters['arterial_resistance']
        rper = self.parameters['peripheral_resistance']

        # Compute the mitral, aortic, and peripheral flowrates.
        q = {'mv': max((pven - pcav)/rven, 0.0),
             'ao': max((pcav - part)/rart, 0.0),
             'per': max((part - 0.0)/rper, 0.0)} # Outflow pressure of aorta is zero (not a closed loop).

        # Compute the LVAD flowrate if this model has an LVAD defined.
        if self._lvad is not None:
            q['lvad'] = self._lvad.compute_flowrate(windkessel_pressures,
                                                    cavity_pressures)['lvad']

        return q

    def compute_pressure(self, windkessel_volumes = None):
        """
        Compute the pressure of the system given a set of windkessel volumes.

        Args:
            windkessel_volumes (optional): Dictionary of windkessel volumes. The current state is used by default.

        Returns:
            Dictionary of computed pressures.
        """
        # Extract the relevant volumes from the inputs.
        if windkessel_volumes is None:
            vart = self.volume['art']
        else:
            vart = windkessel_volumes['art']

        # Extract relevant model parameters.
        cart = self.parameters['arterial_compliance']

        # Compute the arterial pressure. Also return the constant venous pressure.
        p = {'art': vart/cart, 'ven': self.parameters['venous_pressure']}
        return p

    def compute_volume(self, windkessel_pressures = None):
        """
        Compute the volume of the system given a set of windkessel pressures.

        Args:
            windkessel_pressures: Dictionary of windkessel pressures. The current state is used by default.

        Returns:
            Dictionary of computed volumes.
        """
        # Extract the relevant volumes from the inputs.
        if windkessel_pressures is None:
            part = self.pressure['art']
        else:
            part = windkessel_pressures['art']

        # Extract relevant model parameters.
        cart = self.parameters['arterial_compliance']

        # Compute the arterial volume.
        v = {'art': part*cart}
        return v

    @property
    def flowrate(self):
        """
        Return a dictionary of flowrates for the current state.
        """
        q = {'ao': self._qao, 'mv': self._qmv, 'per': self._qper}
        if self._lvad is not None:
            q['lvad'] = self._lvad.flowrate['lvad']
        return q

    @flowrate.setter
    def flowrate(self, values):
        """
        Set the flowrates of the current state from a given dictionary of
        flowrates.
        """
        self._qao = float(values.get('ao', self._qao))
        self._qmv = float(values.get('mv', self._qmv))
        self._qper = float(values.get('per', self._qper))
        if self._lvad is not None:
            q_lvad = values.get('lvad', self._lvad.flowrate['lvad'])
            self._lvad.flowrate = {'lvad': q_lvad}

    @property
    def pressure(self):
        """
        Return a dictionary of pressures for the current state.
        """
        return {'art': self._part, 'ven': self._pven}

    @pressure.setter
    def pressure(self, values):
        """
        Set the pressures of the current state from a given dictionary of
        pressures.
        """
        self._part = float(values.get('art', self._part))
        self._pven = float(values.get('ven', self._pven))

    @property
    def volume(self):
        """
        Return a dictionary of volumes for the current state.
        """
        v = {'art': self._vart, 'ven': self._vven}
        if self._lvad is not None:
            v['lvad'] = self._lvad.volume['lvad']
        return v

    @volume.setter
    def volume(self, values):
        """
        Set the volume of the current state from a given dictionary of volumes.
        """
        self._vart = float(values.get('art', self._vart))
        self._vven = float(values.get('ven', self._vven))

    @property
    def lvad(self):
        """
        Return the attached LVAD object if one exists, else ``None``.
        """
        return self._lvad

    @lvad.setter
    def lvad(self, value):
        """
        Attach an LVAD to this windkessel model.
        """
        self._lvad = value


def get_phase(old_cavity_pressures, windkessel_pressures, cavity_pressures):
    """
    Return the appropriate phase given the old and new system pressures.

    T.H. Adapted to be defined on wk and lv pressures only (with LVAD, some phases may not exist).

    **Phases**:

        1. Filling
        2. Isometric Contraction
        3. Ejection
        4. Isometric Relaxation

    Args:
        old_cavity_pressures: Dictionary of old pressures for the ventricle model.
        windkessel_pressures: Dictionary of pressures for the windkessel model.
        cavity_pressures: Dictionary of pressures for the ventricle model.

    Returns:
        An integer numbered as described above.
    """
    plv_old = old_cavity_pressures['lv']
    plv = cavity_pressures['lv']
    part = windkessel_pressures['art']
    pven = windkessel_pressures['ven']

    if (plv - pven)/pven <= 1e-2 and plv <= part:
        # Note the tolerance (1%): due to volume tolerance, plv may exceed pven, but not significantly.
        return 1  # "I am filling."

    elif plv >= pven and plv >= part:
        return 3  # "I am ejecting."

    # All valves closed: contracting or relaxing.
    elif plv_old < plv:
        return 2  # "I am contracting."

    else:
        return 4  # "I am relaxing."

    # if current_phase == 1 and plv >= pven:
    #     return 2  # "I was filling, now I am contracting."
    #
    # elif current_phase == 2 and plv >= part:
    #     return 3  # "I was contracting, now I am ejecting."
    #
    # elif current_phase == 3 and plv <= part:
    #     return 4  # "I was ejecting, now I am relaxing."
    #
    # elif current_phase == 4 and plv <= pven:
    #     return 1  # "I was relaxing, now I am filling."
    #
    # else:
    #     return current_phase


def get_phase_dc(p_biv_old, p_wk_sys, p_wk_pul, p_biv):
    """
    Return the appropriate phase given the old and new system pressures.
    Generalized for a model with a double circulation, so that part, pven and pcav can be
    systemic, pulmonary, left or right ventricular pressures.

    **Phases**:

        1. Filling
        2. Isometric Contraction
        3. Ejection
        4. Isometric Relaxation

    Args:
        p_biv_old (dict): Old pressures of the ventricles.
        p_wk_sys (dict): Systemic pressures.
        p_wk_pul (dict): Pulmonary pressures.
        p_biv (dict): Pressures of the ventricles.

    Returns:
        Dictionary with phase for 'lv' and 'rv' as described above.
    """
    new_phase = {}

    for key in ['lv', 'rv']:
        if key == 'lv':
            part = p_wk_sys['art']
            pven = p_wk_pul['ven']
        else:
            part = p_wk_pul['art']
            pven = p_wk_sys['ven']

        if (p_biv[key] - pven)/pven <= 1e-2 and p_biv[key] <= part:
            # Note the tolerance (1%): due to volume tolerance, plv may exceed pven, but not significantly.
            new_phase[key] = 1  # "I am filling."

        elif p_biv[key] >= pven and p_biv[key] >= part:
            new_phase[key] = 3  # "I am ejecting."

        # All valves closed: contracting or relaxing.
        elif p_biv_old[key] < p_biv[key]:
            new_phase[key] = 2  # "I am contracting."

        else:
            new_phase[key] = 4  # "I am relaxing."

    return new_phase


def kPa_to_mmHg(p_kPa):
    """
    Convert pressures from kPa to mmHg.

    Args:
        p_kPa: Pressure in kPa.

    Returns:
        Pressure in mmHg.
    """
    conversion_factor = 133.322387415
    return p_kPa*1000/conversion_factor

def mmHg_to_kPa(p_mmHg):
    """
    Convert pressures from mmHg to kPa.

    Args:
        p_mmHg: Pressure in mmHg.

    Returns:
        Pressure in kPa.
    """
    conversion_factor = 133.322387415
    return p_mmHg/1000*conversion_factor