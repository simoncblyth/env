#!/bin/bash 

name=animal
FOLD=/tmp/env/$USER/$name
mkdir -p $FOLD
bin=$FOLD/$name

gcc $name.cc -DWITH_FIX -std=c++17 -Wall -Werror=delete-non-virtual-dtor -lstdc++ -o $bin && $bin

 
