#!/usr/bin/env bash

usage(){ cat << EOU

Setup::

    cd /usr/local/env
    mkdir fastapi_check
    cd fastapi_check

    cp ~/env/fastapi_check/main.py .
    cp ~/env/fastapi_check/dev.sh .

    uv venv
    uv pip install "fastapi[standard]"
    uv pip install numpy



EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

source .venv/bin/activate

which fastapi

fastapi dev main.py




