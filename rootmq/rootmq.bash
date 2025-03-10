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
           for operation without terminal attached :

              startup :
                SCREEN=screen rootmq-sendobj
              detach with : ctrl-A d


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


   == DEBUGGING AMQ WIRING ==

      1) Use rabbitmq-smry + camqadm-- to delete all exchanges/queues (other than the supplied amq.* ones) 
      2) 
          * start producer rootmq-sendobj 
              * creates exchange "abt"

          * start consumer rootmq-pymonitor



   == rootmq updates for rabbitmq-c/codegen tips at Jan 2011 ==

   === example_utils to utils ===  

      http://hg.rabbitmq.com/rabbitmq-c/rev/030b4948b33c
         has become opaque... 
         adopt the new utils.{c,h}

       cp /data/env/local/env/messaging/rmqc/rabbitmq-c/examples/utils.{c,h} .

   ==  amqp_rpc_reply ---> amqp_get_rpc_reply(conn) == 

      a global has become conn local 

   == amqp_exchange_declare has lost the auto_delete param ==


rootmq/src/rootmq.c: In function `rootmq_basic_consume':
rootmq/src/rootmq.c:295: error: too few arguments to function `amqp_basic_consume'




EOU
}

rootmq-dir(){ echo $(env-home)/rootmq ; }
rootmq-cd(){  cd $(rootmq-dir); }
rootmq-mate(){ mate $(rootmq-dir) ; }

rootmq-root-cmd(){  cat << EOC
$SCREEN $(env-runenv) $(which root) -b -q -l $1 $*
EOC
}
rootmq-root(){
   local msg="=== $FUNCNAME :"
   local path=${1:-$defpath}
   [ ! -f "$path" ] && echo $msg no such root script at $path && return 1 
   local cmd=$(rootmq-root-cmd $path)
   echo $msg $cmd 
   eval $cmd 
}

rootmq-iroot(){
   local cmd="$(env-runenv) root -l $*"
   echo $msg $cmd 
   eval $cmd 
}
rootmq-ipython(){
   local cmd="$(env-runenv) $(which ipython) $*"
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


rootmq-groot(){
    cd $(env-libdir)
    echo enter : set args -l $(rootmq-dir)/tests/test_sendobj.C
    echo then  : r 
    gdb $(which root)
}


rootmq-gsendstring(){ rootmq-test mq_sendstring gdb ; }
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
