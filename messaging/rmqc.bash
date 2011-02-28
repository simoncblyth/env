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


   == RabbitMQ Mercurial Usage ==

       http://www.rabbitmq.com/mercurial.html#branchperbug

       http://hgbook.red-bean.com/read/
       http://hgbook.red-bean.com/read/managing-releases-and-branchy-development.html 

            hg log -v -b default  | more

                verbose listing of commits on the default branch 
                ... these are the ones to use 

            "hg parents" is equivalent to "git log"   
                https://github.com/sympy/sympy/wiki/Git-hg-rosetta-stone

            "hg log" gives the entire repo history no matter what



   == fora ==

       http://groups.google.com/group/rabbitmq-discuss/


  == try the tips... ==

      [blyth@cms01 rabbitmq-c]$ hg update -C default
      [blyth@cms01 rabbitmq-codegen]$ hg update -C default


      rmqc-
      rmqc-make   

________Compiling scons-out/dbg/obj/rootmq/src/example_utils.os
rootmq/src/example_utils.c: In function `die_on_amqp_error':
rootmq/src/example_utils.c:33: error: structure has no member named `library_errno'
rootmq/src/example_utils.c:33: error: structure has no member named `library_errno'
scons: *** [scons-out/dbg/obj/rootmq/src/example_utils.os] Error 1
scons: building terminated because of errors.


        http://hg.rabbitmq.com/rabbitmq-c/rev/030b4948b33c
            has become opaque... 


       cp /data/env/local/env/messaging/rmqc/rabbitmq-c/examples/utils.{c,h} .


 
  == getting uptodate ==

      rmqc-cd

      hg branch     ## check are on the default branch 
      hg tip        ## show where the local repository is at ... not necessarily the wc yet

      hg pull       ## pull in updates from remote repository
      hg tip

      hg update     ## update working copy 

 
  == switching to 0_9_1 ==

      http://mercurial.selenic.com/wiki/NamedBranches

      rmqc-
      rmqc-cd
 
      hg branches
      hg update -C amqp_0_9_1     

        ## CAUTION THIS LOOSES UNCOMMITTED CHANGES
        ## .. BUT THESE CHANGES ARE THE RESULT OF rmqc-kludge SO PROBABLY CAN BE APPLIED IN 0_9_1 


[blyth@cms01 rabbitmq-c]$ hg diff
diff -r 277ec3f5b631 configure.ac
--- a/configure.ac      Mon Oct 19 15:17:15 2009 +0100
+++ b/configure.ac      Mon Nov 29 14:15:16 2010 +0800
@@ -21,7 +21,7 @@
 fi
 
 AC_MSG_CHECKING(location of AMQP codegen directory)
-sibling_codegen_dir="$ac_abs_confdir/../rabbitmq-codegen"
+sibling_codegen_dir="/data/env/local/env/messaging/rmqc/rabbitmq-codegen"
 AMQP_CODEGEN_DIR=$(test -d "$sibling_codegen_dir" && echo "$sibling_codegen_dir" || echo "$ac_abs_confdir/codegen")
 AMQP_SPEC_JSON_PATH="$AMQP_CODEGEN_DIR/amqp-0.8.json"
 if test -f "$AMQP_SPEC_JSON_PATH"
diff -r 277ec3f5b631 librabbitmq/amqp.h
--- a/librabbitmq/amqp.h        Mon Oct 19 15:17:15 2009 +0100
+++ b/librabbitmq/amqp.h        Mon Nov 29 14:15:16 2010 +0800
@@ -164,7 +164,7 @@
                              amqp_output_fn_t fn,
                              void *context);
 
-extern int amqp_table_entry_cmp(void const *entry1, void const *entry2);
+extern int amqp_table_entry_cmp(const void *entry1, const void *entry2);
 
 extern int amqp_open_socket(char const *hostname, int portnumber);
 



 ====

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


   == rabbitmqc experience ==

     * after reconfig queue or exchange parameters 
        * delete the exchanges or queues and recreate for change to take effect
        * otherwise gets errors or worse : silent no-errors but config not changed

     * to investifate : google:"rabbitmq-status" rabbitmq plugin (in erlang) providing status web interface 
         * http://www.lshift.net/blog/2009/11/30/introducing-rabbitmq-status-plugin

     * http://groups.google.com/group/rabbitmq-discuss/browse_thread/thread/3835e52e72bd7e87
     
     



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

rmqc-cd(){  cd $(rmqc-dir)/$1; }
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

## caution ... the wrong revision can land you on the wrong branch 
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
   
   rmqc-get- $nam 

   cd $nam
   cmd="hg up $rev"
   echo $msg $cmd ... revert to the pinned revision ... $PWD
   eval $cmd
   cd $iwd
}


rmqc-get-(){
   local msg="$FUNCNAME :"
   local dir=$(dirname $(rmqc-dir)) &&  mkdir -p $dir && cd $dir
   local nam=${1:-$(rmqc-name)}
   [ -d "$nam" ] && echo $msg ABORT dir $nam exists already .. delete and rerun ... sleeping ... ctrl-c to continue  && sleep 1000000
   local cmd="hg clone $(rabbitmq-hg)/$nam"
   echo $msg $cmd ... $PWD
   eval $cmd
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
  #rmqc-kludge

  echo $msg autoreconf  
  autoreconf -i
  echo $msg autoconf  
  autoconf
  echo $msg configure  
  ./configure --prefix=$(rmqc-prefix)

  echo $msg make
  ## avoid hardcoded attempt to use python2.5
  #make PYTHON=python
  make 

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

rmqc-latest(){
   local tmp=/tmp/$USER/env/$FUNCNAME && mkdir -p $tmp && cd $tmp
   hg clone http://hg.rabbitmq.com/rabbitmq-c/

}


