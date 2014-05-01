#!/bin/bash

PREFIX=$VIRTUAL_ENV

make-cmd(){ cat << EOC
cc -I$PREFIX/include -c $name.c && cc -L$PREFIX/lib -lzmq $name.o -o $out && rm $name.o 
EOC
}

make(){
  local name
  for name in $* ; do
     local out=/tmp/$name
     echo $msg making $out
     make-cmd
     eval $(make-cmd)
  done
}

make hwserver hwclient

