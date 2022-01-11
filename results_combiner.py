# Combining Results PDF
# Author: Edmund Dable-Heath
# Simple script for combining the results into a single CSV for ease

import os
from re import findall
import numpy as np
import Theta_Config as config


results = np.zeros((len(os.listdir('Results/')), 2))
for file in os.listdir('Results/'):
    results[int(findall(r'\d+', file)[0])] = np.genfromtxt('Results/'+file, delimiter=',', dtype=None)

np.savetxt('Results/results-dim-'+str(config.dimension)+'-latt-'+str(config.which_latt)+'.csv', results, delimiter=',')

for i in range(len(os.listdir('Results/'))-1):
    if os.path.exists('Results/'+str(i)+'.csv'):
        os.remove('Results/'+str(i)+'.csv')
