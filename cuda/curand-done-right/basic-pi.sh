#!/bin/bash 
usage(){ cat << EOU

~/env/cuda/curand-done-right/basic-pi.sh 

EOU
}

name=basic-pi

cd $(dirname $(realpath $BASH_SOURCE))

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

#opt=--expt-extended-lambda -Wno-deprecated-gpu-targets
opt=

defarg=build_run
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 
    nvcc $name.cu -std=c++11 -I. $opt   -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $bin 1000000
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 


exit 0 

