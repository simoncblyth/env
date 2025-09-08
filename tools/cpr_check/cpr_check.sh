#!/usr/bin/env bash

usage(){ cat << EOU
cpr_check : builds and runs demo code using libcpr C++ binding to libcurl
============================================================================

~/e/tools/cpr_check/cpr_check.sh


* Tested on Darwin using libcurl from macports and cpr built from github sources
  (this worked on new laptop prior to installing Xcode)::


    cd /usr/local/env
    git clone https://github.com/libcpr/cpr.git
    cd cpr && mkdir build && cd build
    cmake .. -DCPR_USE_SYSTEM_CURL=ON
    cmake --build . --parallel
    sudo cmake --install . --prefix /usr/local/e


EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

name=cpr_check
bin=/tmp/$USER/env/tools/$name

mkdir -p $(dirname $bin)


CPR_PREFIX=/usr/local/e


gcc $name.cc -o $bin \
     -Wall \
     -std=c++17 -lstdc++ \
     $(curl-config --cflags) \
     -I$CPR_PREFIX/include \
     $(curl-config --libs) \
     -L$CPR_PREFIX/lib -lcpr
 
[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0



