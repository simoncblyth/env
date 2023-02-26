#!/bin/bash -l 

name=UseTComplex

if [ "$(uname)" == "Darwin" ]; then 
   var=DYLD_LIBRARY_PATH
   ROOT_PREFIX=$HOME/miniconda3
else
   var=LD_LIBRARY_PATH
   #ROOT_PREFIX=$JUNOTOP/ExternalLibs/ROOT/6.22.08
   ROOT_PREFIX=$JUNOTOP/ExternalLibs/ROOT/6.24.06 
fi 

g++ $name.cc \
    -I$HOME/np \
    $(root-config --cflags --ldflags --libs) \
    -o /tmp/$name

[ $? -ne 0 ] && echo compile error && exit 1 


oldruncmd(){ cat << EOC
#$var=$ROOT_PREFIX/lib /tmp/$name   # have to use jre to get a compatble env 
/tmp/$name 
EOC
}

runcmd(){ cat << EOC
/tmp/$name 
EOC
}


runcmd
eval $(runcmd)
[ $? -ne 0 ] && echo run error && exit 2


exit 0 
   

