#!/bin/bash -l 

name=mandelbrot 
defarg="build_run_imshow"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
         -I/usr/local/cuda/include \
         -I/usr/local/opticks/include/OKConf \
         -I/usr/local/opticks/include/SysRap \
         -L/usr/local/opticks/lib \
         -lOKConf \
         -lSysRap \
         -std=c++11 -lstdc++ \
          -o /tmp/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    /tmp/$name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/imshow}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i mandelbrot_imshow.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 1 
fi 



