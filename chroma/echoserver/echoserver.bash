# === func-gen- : chroma/echoserver/echoserver fgp chroma/echoserver/echoserver.bash fgn echoserver fgh chroma/echoserver
echoserver-src(){      echo chroma/echoserver/echoserver.bash ; }
echoserver-source(){   echo ${BASH_SOURCE:-$(env-home)/$(echoserver-src)} ; }
echoserver-vi(){       vi $(echoserver-source) ; }
echoserver-env(){      elocal- ; }
echoserver-usage(){ cat << EOU

ECHOSERVER
===========



EOU
}
echoserver-dir(){ echo $(local-base)/env/chroma/echoserver ; }
echoserver-sdir(){ echo $(env-home)/chroma/echoserver ; }
echoserver-cd(){  cd $(echoserver-dir); }
echoserver-scd(){  cd $(echoserver-sdir); }

echoserver-name(){ echo ZMQEchoServer ; }
echoserver-bin(){ echo /tmp/$(echoserver-name) ; }

echoserver-make(){
  local iwd=$PWD

  echoserver-scd 
  local name=$(echoserver-name) 
  local bin=$(echoserver-bin)

  zeromq-
  cc -I$ZEROMQ_PREFIX/include -c $name.c && cc -L$ZEROMQ_PREFIX/lib -lzmq $name.o -o $bin && rm $name.o 

  ls -l $bin
  cd $iwd
}

echoserver-config(){ echo "tcp://*:5555" ; }
echoserver-export(){ 
   export ECHO_SERVER_CONFIG=$(echoserver-config) 
}

echoserver-run-Linux(){ LD_LIBRARY_PATH=$ZEROMQ_PREFIX/lib ECHO_SERVER_CONFIG=$(echoserver-config) $(echoserver-bin) ; } 
echoserver-run-Darwin(){  echoserver-export ; $(echoserver-bin) ; } 
echoserver-run(){ $FUNCNAME-$(uname) ; } 

echoserver-run-py(){      echoserver-export ;  python $(echoserver-dir)/echoserver.py ; } 

echoserver-gdb(){
  zeromq-
  echoserver-export
  LD_LIBRARY_PATH=$ZEROMQ_PREFIX/lib gdb /tmp/echoserver 
}

echoserver-nuwapkg(){ echo $DYB/NuWa-trunk/dybgaudi/Utilities/ChromaZMQRootTest ; }  
echoserver-nuwapkg-cd(){ cd $(echoserver-nuwapkg) ; }
echoserver-nuwapkg-cpto(){
   local iwd=$PWD

   local src=$(echoserver-nuwapkg)/src 
   cp $(echoserver-name).c $src/

   cd $iwd 
}


