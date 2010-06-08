# === func-gen- : rootmq/rootmq fgp rootmq/rootmq.bash fgn rootmq fgh rootmq
rootmq-src(){      echo rootmq/rootmq.bash ; }
rootmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rootmq-src)} ; }
rootmq-vi(){       vi $(rootmq-source) ; }
rootmq-env(){      
    elocal-  
    root-
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
           run tests using rootmq-root

     rootmq-tests
          run all the tests

    == Usage from python ==

        wiki:Rpath 

     == RUNTIME ERRORS ==

     If the tests (such as rootmq-sendstring) yield ...

Opening socket: Connection refused
ABORT: rootmq_init failed rc : 111 
make: *** [test_sendstring] Error 111

     Check if the node is running, see rabbitmq-usage

EOU
}

rootmq-dir(){ echo $(env-home)/rootmq ; }
rootmq-cd(){  cd $(rootmq-dir); }
rootmq-mate(){ mate $(rootmq-dir) ; }

rootmq-root(){
   local msg="=== $FUNCNAME :"
   local path=${1:-$defpath}
   [ ! -f "$path" ] && echo $msg no such root script at $path && return 1 
   local cmd="$(env-runenv) root -b -q -l $path $*"
   echo $msg $cmd 
   eval $cmd 
}
rootmq-iroot(){
   local cmd="$(env-runenv) root -l $*"
   echo $msg $cmd 
   eval $cmd 
}
rootmq-ipython(){
   local cmd="$(env-runenv) ipython $*"
   echo $msg $cmd 
   eval $cmd 
}

rootmq-chcon-(){
   local lib=$1
   local msg="=== $FUNCNAME :"
   local cmd="sudo chcon -t texrel_shlib_t $(env-libdir)/lib$lib.so"
   echo $msg $cmd
   eval $cmd
}
rootmq-chcon(){
   local libs="rootmq"
   for lib in $libs ; do
       $FUNCNAME- $lib
   done
}

rootmq-tests(){
   rootmq-cd
   local macro
   for macro in tests/*.C ; do
      rootmq-root $macro
   done
}
rootmq-sendobj(){     rootmq-root $(rootmq-dir)/tests/test_sendobj.C ;  }
rootmq-sendjson(){    rootmq-root $(rootmq-dir)/tests/test_sendjson.C ;  }
rootmq-sendstring(){  rootmq-root $(rootmq-dir)/tests/test_sendstring.C ;  }

rootmq-pymonitor(){   rootmq-ipython $(rootmq-dir)/evmq.py $* ; }


rootmq-monitor(){ rootmq-test mq_monitor $* ; }
rootmq-test(){
    local name=${1:-mq_monitor}
    shift

    if [ "$1" == "gdb" ]; then
       cd $(env-libdir)
       echo $msg workaround gdb env wierdness by running from $PWD
    fi

    local cmd="env -i HOME=$HOME ABERDEEN_HOME=$ABERDEEN_HOME ENV_PRIVATE_PATH=$ENV_PRIVATE_PATH LD_LIBRARY_PATH=$(env-libdir) $* $(env-testsdir)/$name"
    echo $msg $cmd
    eval $cmd
}
rootmq-monitor-term(){ rootmq-term mq_monitor ; }
rootmq-term(){
    local name=${1:-mq_monitor}
    local cmd="kill -TERM $(pgrep $name)"
    echo $msg $cmd
    eval $cmd
}
