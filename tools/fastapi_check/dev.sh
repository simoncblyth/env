#!/usr/bin/env bash

usage(){ cat << EOU
dev.sh
========

This uses the *uv* python package+venv tool::

    https://github.com/astral-sh/uv

Build and start the endpoint::

    ~/env/tools/fastapi_check/dev.sh

Make HTTP POST requests to the endpoint::

     ~/np/tests/np_curl_test/np_curl_test.sh
     LEVEL=1 MULTIPART=0  ~/np/tests/np_curl_test/np_curl_test.sh
     LEVEL=1 MULTIPART=1  ~/np/tests/np_curl_test/np_curl_test.sh

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

defarg="info_venv_run"
arg=${1:-$defarg}


if [ "${arg/info}" != "$arg" ]; then
    vv="BASH_SOURCE PWD defarg arg"
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/venv}" != "$arg" ]; then
    if [ ! -d ".venv" ]; then
        echo $BASH_SOURCE - installing dependencies
        echo .venv > .gitignore
        echo __pycache__ >> .gitignore
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
    fastapi dev main.py
    [ $? -ne 0 ] && echo $BASH_SOURCE - failed to fastapi dev && exit 2
fi

exit 0
