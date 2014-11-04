#!/usr/bin/env python
import numpy as np
import sys

if __name__ == '__main__':
    for path in sys.argv[1:]:
        a = np.load(path)
        print "\n",path,"\n",a.dtype,"\n",a.shape,"\n",a

