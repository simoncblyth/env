#!/usr/bin/env bash 

usage(){ cat << EOU
nanobind_check.sh
===================

On zeta initial checks done in /usr/local/env/nanobind_check


::

    cd ~/env/tools/nanobind_check

    ./nanobind_check.sh venv
    ./nanobind_check.sh info
    ./nanobind_check.sh build
    ./nanobind_check.sh pdb


EOU
}


name=nanobind_check
script=$name.py


defarg="venv_info_build_pdb"
arg=${1:-$defarg}

vv="BASH_SOURCE PWD"

cd $(dirname $(realpath $BASH_SOURCE))

if [ "${arg/venv}" != "$arg" ]; then
    if [ ! -d ".venv" ]; then
        uv venv
        uv pip install nanobind numpy ipython
    else
         echo $BASH_SOURCE .venv exists already 
    fi
    source .venv/bin/activate
fi


if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done 

    which cmake
    which python
    which python3
    which ipython

    cmake --version
    python --version
    python3 --version
    ipython --version
fi 


if [ "${arg/build}" != "$arg" ]; then

    if [ ! -d "build" ]; then
        cmake -S . -B build 
        [ $? -ne 0 ] && echo $BASH_SOURCE - config error && exit 1
    fi  
    cmake --build build 
    [ $? -ne 0 ] && echo $BASH_SOURCE - build error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
   PYTHONPATH=build ipython --pdb -i $script
fi 

exit 0

