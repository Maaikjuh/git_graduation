import os

# --------------------------------------------------------------------------- #
# Specify paths of relevant directories.
# --------------------------------------------------------------------------- #
# Root directory (for easily specifying paths).
#ROOT_DIRECTORY = r'E:\Graduation project'
# Automatically deduce root directory from the path to this file:
ROOT_DIRECTORY = os.path.join(
        os.path.abspath(__file__).split('\\Graduation_project\\')[0],
        'Graduation_project')
path = os.path.abspath(__file__)
split_path = os.path.abspath(__file__).split('\\Graduation_project\\')[0]

SIMULATION_DATA_PATH = os.path.join(split_path, 'Graduation_project\Results_Tim\systemic_circulation_27_02')
CVBTK_PATH = os.path.join(ROOT_DIRECTORY, 'Tim_Hermans\model\cvbtk')

import sys
hemodyn= sys.path.append(r'C:\Users\Maaike\Documents\Master\Graduation_project\git_graduation_project\cvbtk')

print('{}'.format(ROOT_DIRECTORY)) 
#print('{}'.format(path))
print('{}'.format(split_path))
print('{}'.format(SIMULATION_DATA_PATH))
#print('{}'.format(CVBTK_PATH))
#print('{}'.format(hemodyn))

check_paths = [ROOT_DIRECTORY,
#               LIFETEC_DATA_PATH,
               SIMULATION_DATA_PATH,
               CVBTK_PATH]
  #             POSTPROCESSING_BIV_PATH]
for path in check_paths:
    if not os.path.exists(path):
        #print("nope")
        x=1