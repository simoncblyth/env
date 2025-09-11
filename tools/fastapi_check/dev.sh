#!/usr/bin/env bash

usage(){ cat << EOU
dev.sh
======

This uses the *uv* python package+venv tool::

    https://github.com/astral-sh/uv

Initial setup, create dir for uv .venv and plant symbolic link::

    mkdir /usr/local/env/fastapi_check
    cd /usr/local/env/fastapi_check
    ln -s ~/env/tools/fastapi_check/dev.sh
    /usr/local/env/fastapi_check/dev.sh info_venv

Start the endpoint::

    /usr/local/env/fastapi_check/dev.sh

Make requests::

     ~/np/tests/np_curl_test/np_curl_test.sh
     LEVEL=1 MULTIPART=0  ~/np/tests/np_curl_test/np_curl_test.sh
     LEVEL=1 MULTIPART=1  ~/np/tests/np_curl_test/np_curl_test.sh


VDIR
    unresolved symbolic link, giving directory containing uv .venv eg:  /usr/local/env/fastapi_check

SDIR
    resolved symbolic link, giving source directory eg: ~/env/tools/fastapi_check

EOU
}


VDIR=$(dirname $BASH_SOURCE)
SDIR=$(dirname $(realpath $BASH_SOURCE))
cd $VDIR

defarg="info_venv_run"
arg=${1:-$defarg}


if [ "${arg/info}" != "$arg" ]; then
    vv="BASH_SOURCE PWD VDIR SDIR"
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/venv}" != "$arg" ]; then
    if [ ! -d ".venv" ]; then
        echo $BASH_SOURCE - installing dependencies
        uv venv
        uv pip install "fastapi[standard]" numpy
    else
        echo $BASH_SOURCE - using existing .venv
    fi
fi

if [ "${arg/run}" != "$arg" ]; then

    source .venv/bin/activate
    [ $? -ne 0 ] && echo $BASH_SOURCE - failed to activate venv && exit 1

    which fastapi
    fastapi dev $SDIR/main.py
    [ $? -ne 0 ] && echo $BASH_SOURCE - failed to fastapi dev && exit 2
fi

exit 0
