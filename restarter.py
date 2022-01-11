# Restarting the program if paused
# Author: Edmund Dable-Heath
# This will generate the remaining values left to be tested as an iterable so the concurrency can be dealt with if the
# job gets paused as a sub job.

from os import walk, listdir
from re import findall
from itertools import filterfalse
import numpy as np


(_, _, names) = next(walk('Results/'))
print(names)
for i in names:
    print(names)
    print(findall(r'\d+', i))
    print(findall(r'\d+', i)[0])
    print(type(findall(r'\d+', i)[0]))
