htdocs-src(){    echo trac/htdocs.bash ; }
htdocs-source(){ echo ${BASH_SOURCE:-$(env-home)/$(htdocs-src)} ; }
htdocs-vi(){ vi $BASH_SOURCE ; }
htdocs-env(){ trac- ; }

htdocs-usage(){
  cat << EOU

      htdocs-url : $(htdocs-url)
      htdocs-dir : $(htdocs-dir)
      htdocs-rpath <path-to-local-file> <htdocs-relative-path-on-server> 
           path on the server
 
          eg 
          htdocs-rpath $(htdocs-source) an/example/shim
             -->   $(htdocs-rpath $(htdocs-source) an/example/shim)
  

      env-designated : $(env-designated)
      TRAC_INSTANCE : $TRAC_INSTANCE

      htdocs-up  <path> <on-server-htdocs-rel-path>
          usage examples :
             TRAC_INSTANCE=heprez htdocs-up $(htdocs-source)
             TRAC_INSTANCE=heprez htdocs-up $(htdocs-source) $(htdocs-privat)

      Download with 
          curl -O  $(TRAC_INSTANCE=heprez htdocs-url)/on/server/relative/name.whatever

      htdocs-test

      htdocs-privat 
           the contents are not listable but nevertheless some mimimal
           protection can be advantageous sometimes 


EOU

}


htdocs-url(){       echo $(env-localserver)/tracs/$TRAC_INSTANCE/chrome/site ; }
htdocs-dir(){       echo $(local-scm-fold $(env-designated))/tracs/$TRAC_INSTANCE/htdocs ; }

htdocs-rpath(){     
   local path=$1
   local rel=$2    
   [ -n "$rel" ] && rel="$rel/" 
   echo $(htdocs-dir)/$rel$(basename $path) 
}

htdocs-up(){    
   local msg="=== $FUNCNAME :"
   local path=$1
   local rpath=$(htdocs-rpath $*)
   echo $msg  $path to $(env-designated):$rpath 

   cat $path | ssh $(env-designated) "mkdir -p $(dirname $rpath) && cat - > $rpath "     
}

htdocs-privat(){ echo $(private- ; private-val HTDOCS_PRIVAT) ; }

htdocs-test(){

   local msg="=== $FUNCNAME :"
   local path=$(htdocs-source)
   local name=$(basename $path)
   local ins=heprez
   local rel=on/server/htdocs/relative
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp

   TRAC_INSTANCE=$ins htdocs-upcat $path $rel 
   curl -o $tmp/$name $(TRAC_INSTANCE=$ins htdocs-url)/$rel/$name
   diff $path $tmp/$name 
  
}


