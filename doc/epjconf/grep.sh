#!/usr/bin/env bash

cd $(dirname $(realpath $BASH_SOURCE))

search=${1:-genstep}

grep -r $search .


