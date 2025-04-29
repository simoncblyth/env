#!/bin/bash 
usage(){ cat << EOU

::

    hookup_conda_ok

    source chararraytest.sh




EOU
}


f2py -m Fortran -c chararraytest.f90

PYTHONPATH=. python test_chararray.py 


