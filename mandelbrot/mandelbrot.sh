#!/bin/bash -l 
usage(){ cat << EOU
mandelbrot.sh
===============

FOCUS=-1.45,0,0.05 ~/env/mandelbrot/mandelbrot.sh
FOCUS=-1.45,0,0.05 MIT=80 ~/env/mandelbrot/mandelbrot.sh
FOCUS=-1.45,0,0.05 MIT=50 ~/env/mandelbrot/mandelbrot.sh

EOU
}
cd $(dirname $BASH_SOURCE)
name=mandelbrot 
defarg="build_run_ana"
arg=${1:-$defarg}

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -I$HOME/np -std=c++11 -lstdc++ -o $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 
if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 
if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 1 
fi 
exit 0 
