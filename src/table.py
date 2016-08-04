from tabulate import tabulate
import numpy as np

print tabulate(np.random.rand(5,3), floatfmt='.2f')
