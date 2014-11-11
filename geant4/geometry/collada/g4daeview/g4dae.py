#!/usr/bin/env python
import os
import numpy as np
 
ph = lambda _:np.load(os.environ['DAE_PATH_TEMPLATE'] % _)


def main():
    import sys
    np.set_printoptions(precision=3,suppress=True)

    name = sys.argv[1]
    a = ph(name)
    print a

    if len(sys.argv) > 2 and sys.argv[2] == '-i': 
        import IPython
        IPython.embed() 


if __name__ == '__main__':
    main()



