# === func-gen- : notifymq/notifymq fgp notifymq/notifymq.bash fgn notifymq fgh notifymq
notifymq-src(){      echo notifymq/notifymq.bash ; }
notifymq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(notifymq-src)} ; }
notifymq-vi(){       vi $(notifymq-source) ; }
notifymq-env(){      elocal- ; }
notifymq-usage(){
  cat << EOU
     notifymq-src : $(notifymq-src)
     notifymq-dir : $(notifymq-dir)





EOU
}

notifymq-preq(){

   rabbitmq-c-get


}



notifymq-dir(){ echo $(env-home)/notifymq ; }
notifymq-cd(){  cd $(notifymq-dir); }
notifymq-mate(){ mate $(notifymq-dir) ; }
notifymq-get(){
   local dir=$(dirname $(notifymq-dir)) &&  mkdir -p $dir && cd $dir

}



notifymq-root(){
   local msg="=== $FUNCNAME :"
   local defpath=$(notifymq-dir)/tests/test_basic_consume.C
   local path=${1:-$defpath}
   [ ! -f "$path" ] && echo $msg no such root script at $path && return 1 
 
   rabbitmq-
   cjson-
   priv-
   local cmd="LD_LIBRARY_PATH=$(rabbitmq-c-libdir):$(cjson-libdir):$(priv-libdir) root -q -l $path"
   echo $msg $cmd
   eval $cmd

}
