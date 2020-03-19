"""
Calls postprocessing functions for FE simulations.

Copy this script to the output directory and run it.
"""

from cvbtk import Export

# Specify directory.
directory = '.'

# Specify cycle to postprocess.
# If None, the last available cycle is postprocessed.
cycle = None

# Specify the reference state for the strain.
# Choose between 'stress_free' (reference is the stress-free state), or 
# 'begin_ic' (reference is the state at onset of LV isovolumic contraction phase).
strain_reference = 'stress_free'

print('strain reference: {}'.format(strain_reference))

# For additional options and their default settings
# (e.g. you can choose the strain reference state),
# see the documentation in the Export class.
Export(directory, cycle=cycle, strain_reference=strain_reference)
