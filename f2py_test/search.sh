#!/bin/bash
usage(){ cat << EOU
search.sh
===========

* https://stackoverflow.com/questions/7632963/numpy-find-first-index-of-value-fast
* https://gist.github.com/mverleg/71dbbf755ed1cf96589236a69bc64027
* https://gist.github.com/mverleg/3b63d70c806ea4eb73168870297d0bcb


Prior to using this hookup conda and activate the 
virtual enviroment used for analysis, eg::

    hookup_conda       ## non-standard bash function fished out of .bash_profile
    conda activate ok  ## ok is environment with matplotlib+pyvista+meson(for f2py)

Use "source" in order to invoke this with the f2py/meson 
from that analysis python environment::

    source search.sh 

Note that as it is likely that the build environment python/numpy
will be a lot older and with fewer modules than the analysis environment 
it will typically not be productive to integrate this with CMake. 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
which f2py

if [ -n "$SEPARATED_BUILD" ]; then 

    echo [ build pyf signature file
    f2py -m search -h search.pyf --overwrite-signature  search.f90
    echo ] build pyf signature file

    echo [ compile the module
    f2py -c search.pyf search.f90
    echo ] compile the module

else
    echo [ hole-in-one compile the module
    f2py -c --backend meson -m search search.f90
    echo ] hole-in-one compile the module
fi 


echo [ install the module
mod=$(ls -1 search.cpython*.so)
dest=$OPTICKS_PREFIX/py/opticks/ana
if [ -d "$dest" ]; then
   cmd="mv $mod $dest/"
   echo $cmd
   eval $cmd
fi 
ls -alst $dest/search*
echo ] install the module

echo [ test module
PYTHONPATH=$OPTICKS_PREFIX/py python search_test.py
echo ] test module


