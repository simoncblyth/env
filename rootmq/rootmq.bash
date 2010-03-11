# === func-gen- : rootmq/rootmq fgp rootmq/rootmq.bash fgn rootmq fgh rootmq
rootmq-src(){      echo rootmq/rootmq.bash ; }
rootmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rootmq-src)} ; }
rootmq-vi(){       vi $(rootmq-source) ; }
rootmq-env(){      
    elocal-  
    root-
    rabbitmq-
    cjson-
    priv-
    aberdeen-
}
rootmq-usage(){
  cat << EOU
     rootmq-src : $(rootmq-src)
     rootmq-dir : $(rootmq-dir)

     rootmq-root path/to/macro.C
           batch run macro with dependent libpaths defined

     rootmq-iroot 
     rootmq-ipython
           interactive root/python with libpaths setup appropriately 


     rootmq-sendstring
     rootmq-sendjson
     rootmq-sendobj
           run tests via the Makefile which is reponsible for environment control for library access


     rootmq-monitor 



     == RUNTIME ERRORS ==

     If the tests (such as rootmq-sendstring) yield ...

Opening socket: Connection refused
ABORT: rootmq_init failed rc : 111 
make: *** [test_sendstring] Error 111

     Check if the node is running, see rabbitmq-usage
     




EOU
}

rootmq-preq(){
   rabbitmq-c-get
}
rootmq-dir(){ echo $(env-home)/rootmq ; }
rootmq-libdir(){ echo $(rootmq-dir)/lib ; }
rootmq-cd(){  cd $(rootmq-dir); }
rootmq-mate(){ mate $(rootmq-dir) ; }

rootmq-libpaths(){ echo $(rootmq-libdir):$(rabbitmq-c-libdir):$(cjson-libdir):$(priv-libdir):$(aberdeen-libdir) ; }
rootmq-dynpaths(){  
  case $(uname) in 
      Darwin) echo DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(rootmq-libpaths) ;;
       Linux) echo LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(rootmq-libpaths) ;;
  esac
}

rootmq-root(){
   local msg="=== $FUNCNAME :"
   local path=${1:-$defpath}
   [ ! -f "$path" ] && echo $msg no such root script at $path && return 1 
   local cmd="$(rootmq-dynpaths) root -q -l $path $*"
   echo $msg $cmd 
   eval $cmd 
}
rootmq-iroot(){
   local cmd="$(rootmq-dynpaths) root -l $*"
   echo $msg $cmd 
   eval $cmd 
}
rootmq-ipython(){
   local cmd="$(rootmq-dynpaths) ipython $*"
   echo $msg $cmd 
   eval $cmd 
}


rootmq-chcon(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo chcon -t texrel_shlib_t $(rootmq-libdir)/librootmq.so"
   echo $msg $cmd
   eval $cmd
}


rootmq-make(){       rootmq-cd ; make $* ; }
rootmq-sendstring(){ rootmq-make test_sendstring ; }
rootmq-sendjson(){   rootmq-make test_sendjson   ; }
rootmq-sendobj(){    rootmq-make test_sendobj    ; }



