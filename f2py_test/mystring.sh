#!/bin/bash
usage(){ cat << EOU

https://stackoverflow.com/questions/41864984/how-to-pass-array-of-strings-to-fortran-subroutine-using-f2py

::

    hookup_conda_ok

    source mystring.sh 

    source mystring.sh  pdb 

EOU
}

defarg="build_pdb"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then
   f2py -m mystring -c mystring.f90
   [ $? -ne 0 ] && echo $BASH_SOURCE build error 
fi 

if [ "${arg/pdb}" != "$arg" ]; then
   PYTHONPATH=. ipython --pdb -i mystring_test.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE pdb error 
fi

