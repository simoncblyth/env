#!/bin/bash -l 

name=UseTComplex

if [ "$(uname)" == "Darwin" ]; then 
   var=DYLD_LIBRARY_PATH
   ROOT_PREFIX=$HOME/miniconda3
else
   var=LD_LIBRARY_PATH
   ROOT_PREFIX=$JUNOTOP/ExternalLibs/ROOT/6.22.08
fi 

gcc $name.cc \
    -std=c++11 \
    -I$HOME/np \
    -I$ROOT_PREFIX/include \
    -L$ROOT_PREFIX/lib \
    -lstdc++ \
    -lCore \
    -lHist \
    -lMathCore \
    -o /tmp/$name

[ $? -ne 0 ] && echo compile error && exit 1 


runcmd(){ cat << EOC
$var=$ROOT_PREFIX/lib /tmp/$name
EOC
}


runcmd
eval $(runcmd)
[ $? -ne 0 ] && echo run error && exit 2


exit 0 
   

