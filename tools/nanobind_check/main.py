#!/usr/bin/env python

import numpy as np
import my_ext
#print(help(my_ext))


def main():

    a0 = np.ones([4,4,3], dtype=np.uint8 )
    a = a0.copy()
    a3 = a0.copy()

    my_ext.process(a)
    my_ext.process3(a3)


    print("a0\n",a0)
    print("a\n",a)
    assert( np.all( a0*2 == a ))


    print("a3\n",a3)
    assert( np.all( a0*3 == a3 ))


    b = my_ext.create_2d(3,3)
    print("b\n", b)


if __name__ == '__main__':
    main()
pass

