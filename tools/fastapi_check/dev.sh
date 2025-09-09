#!/usr/bin/env bash

usage(){ cat << EOU

Setup, create "fastapi_check" directory for the .venv to live in::

    cd /usr/local/env
    mkdir fastapi_check
    cd fastapi_check


Activate the venv and invoke "fastapi dev" with the main.py::

    ~/env/tools/fastapi_check/dev.sh


EOU
}

SDIR=$(dirname $(realpath $BASH_SOURCE))

vv="BASH_SOURCE SDIR PWD"
for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done


if [ ! -d ".venv" ]; then
    echo $BASH_SOURCE - installing dependencies 
    uv venv
    uv pip install "fastapi[standard]" numpy
else
    echo $BASH_SOURCE - using existing .venv
fi 

source .venv/bin/activate

which fastapi

fastapi dev $SDIR/main.py



