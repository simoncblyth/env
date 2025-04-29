#!/bin/bash
usage(){ cat << EOU

https://stackoverflow.com/questions/41864984/how-to-pass-array-of-strings-to-fortran-subroutine-using-f2py

::

    hookup_conda_ok

    source idxstring.sh 

    source idxstring.sh  pdb 

EOU
}

defarg="build_pdb"
arg=${1:-$defarg}

name=idxstring

if [ "${arg/build}" != "$arg" ]; then
   f2py -m $name -c $name.f90
   [ $? -ne 0 ] && echo $BASH_SOURCE build error 
fi 

if [ "${arg/pdb}" != "$arg" ]; then
   PYTHONPATH=. ipython --pdb -i ${name}_test.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE pdb error 
fi

