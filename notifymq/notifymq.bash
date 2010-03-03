# === func-gen- : notifymq/notifymq fgp notifymq/notifymq.bash fgn notifymq fgh notifymq
notifymq-src(){      echo notifymq/notifymq.bash ; }
notifymq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(notifymq-src)} ; }
notifymq-vi(){       vi $(notifymq-source) ; }
notifymq-env(){      
    elocal-  
    root-
    rabbitmq-
    cjson-
    #priv-
    aberdeen-
}
notifymq-usage(){
  cat << EOU
     notifymq-src : $(notifymq-src)
     notifymq-dir : $(notifymq-dir)

     notifymq-root path/to/macro.C
           batch run macro with dependent libpaths defined

     notifymq-iroot 
     notifymq-ipython
           interactive root/python with libpaths setup appropriately 


     notifymq-sendstring
     notifymq-sendjson
     notifymq-sendevt
           run tests via the Makefile which is reponsible for environment control for library access


     notifymq-monitor 



     == RUNTIME ERRORS ==

     If the tests (such as notifymq-sendstring) yield ...

Opening socket: Connection refused
ABORT: notifymq_init failed rc : 111 
make: *** [test_sendstring] Error 111

     Check if the node is running, see rabbitmq-usage
     




EOU
}

notifymq-preq(){
   rabbitmq-c-get
}
notifymq-dir(){ echo $(env-home)/notifymq ; }
notifymq-libdir(){ echo $(notifymq-dir)/lib ; }
notifymq-cd(){  cd $(notifymq-dir); }
notifymq-mate(){ mate $(notifymq-dir) ; }

notifymq-libpaths(){ echo $(notifymq-libdir):$(rabbitmq-c-libdir):$(cjson-libdir):$(priv-libdir):$(aberdeen-libdir) ; }
notifymq-dynpaths(){  
  case $(uname) in 
      Darwin) echo DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$(notifymq-libpaths) ;;
       Linux) echo LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(notifymq-libpaths) ;;
  esac
}


notifymq-root(){
   local msg="=== $FUNCNAME :"
   local path=${1:-$defpath}
   [ ! -f "$path" ] && echo $msg no such root script at $path && return 1 
   local cmd="$(notifymq-dynpaths) root -q -l $path $*"
   echo $msg $cmd 
   eval $cmd 
}
notifymq-iroot(){
   local cmd="$(notifymq-dynpaths) root -l $*"
   echo $msg $cmd 
   eval $cmd 
}
notifymq-ipython(){
   local cmd="$(notifymq-dynpaths) ipython $*"
   echo $msg $cmd 
   eval $cmd 
}




notifymq-chcon(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo chcon -t texrel_shlib_t $(notifymq-libdir)/libnotifymq.so"
   echo $msg $cmd
   eval $cmd
}

notifymq-sendstring(){
   notifymq-cd
   make test_sendstring
}

notifymq-sendjson(){
   notifymq-cd
   make test_root2cjson
}

notifymq-sendevt(){
   notifymq-cd
   make test_root2message
}



notifymq-monitor(){
   notifymq-cd
   make test_monitor
}
notifymq-gmonitor(){
   notifymq-cd
   make test_gmonitor
}
notifymq-build(){
   notifymq-cd
   make

}


