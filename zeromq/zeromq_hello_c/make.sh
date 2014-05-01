#!/bin/bash -l

zeromq-
PREFIX=$(zeromq-prefix)

hello-make-cmd(){ cat << EOC
cc -I$PREFIX/include -c $name.c && cc -L$PREFIX/lib -lzmq $name.o -o $out && rm $name.o 
EOC
}

hello-make(){
  local name
  for name in $* ; do
     local out=/tmp/$name
     echo $msg making $out
     hello-make-cmd
     eval $(hello-make-cmd)
  done
}

hello-make hwserver hwclient

