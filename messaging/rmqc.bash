# === func-gen- : messaging/rmqc fgp messaging/rmqc.bash fgn rmqc fgh messaging
rmqc-src(){      echo messaging/rmqc.bash ; }
rmqc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rmqc-src)} ; }
rmqc-vi(){       vi $(rmqc-source) ; }
rmqc-env(){      
   elocal-  
   rabbitmq-
}
rmqc-usage(){
  cat << EOU
     rmqc-src : $(rmqc-src)
     rmqc-dir : $(rmqc-dir)

   This seeks to replace the rabbitmq-c-* functions housed in rabbitmq- 
   to avoid building confusions 

   The env-config triplet ..
       rmqc-libname : $(rmqc-libname)
       rmqc-libdir  : $(rmqc-libdir)
       rmqc-incdir  : $(rmqc-incdir)

   Which exports the config and is published by e/bin/env-config script
   or function : env-config 
       env-config rmqc --libdir


       rmqc-rev : $(rmqc-rev)
           revision at which rabbitmq-c is pinned, check the working
           copy using : "hg id"

       rmqc-codegen-rev  : $(rmqc-codegen-rev)
           


       rmqc-wipe
           remove the build dir and examples

       rmqc-build
           download configure make install



EOU
}

# env-config exports
rmqc-libname(){ echo rabbitmq ; }
rmqc-libdir(){  echo $(rmqc-prefix)/lib ; }
rmqc-incdir(){  echo $(rmqc-prefix)/include ; }
# pkgconfig exports
rmqc-pkgconfig-(){  cat << EOM
prefix=$(rmqc-prefix)
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${exec_prefix}/include
somevar=simon
 
Name: $(rmqc-name) 
Description: $(rmqc-desc)
Version: $(rmqc-version)
Libs: -L\${libdir} -lrabbitmq  -Wl,-rpath,\${libdir}
Cflags: -I\${includedir}
EOM
}

rmqc-pkg(){    echo rmqc ; }
rmqc-pkgconfig(){
  pkgconfig-
  $FUNCNAME- | pkgconfig-plus ${1:-$(rmqc-pkg)}
}



rmqc-desc(){ echo Network Client to RabbitMQ Server implemented in C ; }
rmqc-name(){ echo rabbitmq-c ; }
rmqc-prefix(){ echo $(local-base)/env/rmqc ; }
rmqc-dir(){    echo $(local-base)/env/messaging/rmqc/$(rmqc-name) ; }
rmqc-base(){   echo $(dirname $(rmqc-dir)) ; }
rmqc-codegen-dir(){ echo $(rmqc-base)/rabbitmq-codegen ; }

rmqc-cd(){  cd $(rmqc-dir); }
rmqc-codegen-cd(){  cd $(rmqc-codegen-dir) ; }
rmqc-mate(){ mate $(rmqc-dir) ; }

rmqc-wipe(){
   local msg="=== $FUNCNAME :"
   local dir=$(rmqc-base)
   cd $(dirname $dir)
   local cmd="rm -rf $dir"
   read -p "$msg enter YES to proceed with : $cmd "  ans
   [ "$ans" != "YES" ] && echo $msg skipping && return 1
   eval $cmd
}

rmqc-build(){
   rmqc-wipe
   rmqc-preq
   rmqc-get
   rmqc-codegen-get

   rmqc-make
}


rmqc-version(){     echo dev ; }
rmqc-rev(){         echo 277ec3f5b631 ; }
rmqc-codegen-rev(){ echo 821f5ee7b040 ; }

rmqc-preq(){
   pip install simplejson 
}


rmqc-headers(){
  find $(rmqc-base) -name amqp.h -exec ls -l {} \;
  find $(rmqc-prefix) -name amqp.h -exec ls -l {} \;

}

rmqc-pinclone(){
   local msg="=== $FUNCNAME :"
   local nam=$1
   local rev=$2
   local iwd=$PWD
   
   [ -d "$nam" ] && echo $msg ABORT dir $nam exists already .. delete and rerun ... sleeping ... ctrl-c to continue  && sleep 1000000
   local cmd="hg clone $(rabbitmq-hg)/$nam"
   echo $msg $cmd ... $PWD
   eval $cmd
  
   cd $nam
   cmd="hg up $rev"
   echo $msg $cmd ... revert to the pinned revision ... $PWD
   eval $cmd
   cd $iwd
}



rmqc-get(){
   local msg="$FUNCNAME :"
   local dir=$(dirname $(rmqc-dir)) &&  mkdir -p $dir && cd $dir
   rmqc-pinclone $(rmqc-name) $(rmqc-rev)
}

rmqc-codegen-get(){
  local msg="=== $FUNCNAME :"
  local dir=$(dirname $(rmqc-codegen-dir))
  local nam=$(basename $(rmqc-codegen-dir))   ## rabbitmq-codegen
  mkdir -p $dir && cd $dir
  rmqc-pinclone $nam $(rmqc-codegen-rev)
}

rmqc-kludge(){
  local msg="=== $FUNCNAME :"
  echo $msg from $PWD
  perl -pi -e "s,(sibling_codegen_dir=).*,\$1\"$(rmqc-codegen-dir)\"," configure.ac
  perl -pi -e 's,void const,const void,g' librabbitmq/amqp.h   ## needed to get past rootcint in notifymq build
}


rmqc-make(){
  local msg="=== $FUNCNAME :"
  rmqc-cd
  rmqc-kludge

  echo $msg autoreconf  
  autoreconf -i
  echo $msg autoconf  
  autoconf
  echo $msg configure  
  ./configure --prefix=$(rmqc-prefix)

  echo $msg make
  ## avoid hardcoded attempt to use python2.5
  make PYTHON=python

  echo $msg install
  make install

  rmqc-pkgconfig 
}





######### TESTING 

rmqc-exepath(){  echo $(rmqc-dir)/examples/amqp_$1 ; }
rmqc-exchange(){ echo ${RMQC_EXCHANGE:-amq.direct} ; }
rmqc-queue(){    echo ${RMQC_QUEUE:-test queue} ; }
rmqc-key(){      echo ${RMQC_KEY:-test queue} ; }
rmqc-consumer(){
   local msg="=== $FUNCNAME :"
   local exe=$(rmqc-exepath consumer)
   [ ! -x "$exe" ] && echo $msg ABORT no executable at $exe && return 1 
   private-
   local host=$(private-val AMQP_SERVER) 
   local port=$(private-val AMQP_PORT) 
   local cmd="$exe $host $port " 
   echo $msg $cmd CAUTION hardcoded : exchange  \"amq.direct\" and key  \"test queue\"
   eval $cmd
}

rmqc-sendstring(){
   local msg="=== $FUNCNAME :"
   local exe=$(rmqc-exepath sendstring)
   [ ! -x "$exe" ] && echo $msg ABORT no executable at $exe && return 1 
   
   private-
   local host=$(private-val AMQP_SERVER) 
   local port=$(private-val AMQP_PORT) 
   local exchange=$(rmqc-exchange)
   local routingkey=$(rmqc-queue)
   local messagebody="$(hostname) $(date)"
   local cmd="$exe $host $port $exchange \"$routingkey\" \"$messagebody\""
   echo $msg $cmd
   eval $cmd
}

rmqc-listen(){
   local msg="=== $FUNCNAME :"
   local exe=$(rmqc-exepath listen)
   [ ! -x "$exe" ] && echo $msg ABORT no executable at $exe && return 1 
   
   private-
   local host=$(private-val AMQP_SERVER) 
   local port=$(private-val AMQP_PORT) 
   local exchange=$(rmqc-exchange)
   local routingkey=$(rmqc-queue)
   local cmd="$exe $host $port $exchange \"$routingkey\" "
   echo $msg $cmd
   eval $cmd

}

rmqc-usage(){
   local files="amqp_sendstring.c example_utils.c example_utils.h"
   cd $(env-home)/notifymq 
   local file ; for file in $files ; do
     [ ! -f "$file" ] &&  cp $(rabbitmq-c-dir)/examples/$file . || echo $msg $file already present in $PWD
   done
}



