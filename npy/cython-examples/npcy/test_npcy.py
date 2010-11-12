
import numpy as np
from npcy import myfunc


a = np.arange(40,dtype=np.float64).reshape((4,10))   ## array has to be shaped to fix the func ...   n * 2n
myfunc(a)

