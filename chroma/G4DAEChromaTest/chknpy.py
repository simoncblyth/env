#!/usr/bin/env python
import sys, os
import numpy as np

def main():
    np.set_printoptions(precision=3, suppress=True)
    path = os.environ['DAE_PATH_TEMPLATE_NPY'] % sys.argv[1]
    a = np.load(path)
    print path
    print a
    print a.shape
    print a.dtype


if __name__ == '__main__':
    main()
