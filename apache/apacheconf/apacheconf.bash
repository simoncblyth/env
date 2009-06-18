
apacheconf-src(){ echo apache/apacheconf/apacheconf.bash ; }
apacheconf-source(){ echo $ENV_HOME/$(apacheconf-src) ; }
apacheconf-vi(){    vi $(apacheconf-source) ; }

apacheconf-usage(){
  
   cat << EOU

     apacheconf-src          : $(apacheconf-src)
     apacheconf-svi

     apacheconf-dir          :  $(apacheconf-dir)
     apacheconf-path         :  $(apacheconf-path)
     apacheconf-patchpath    :  $(apacheconf-patchpath)
     apacheconf-envvars-path :  $(apacheconf-envvars-path)
   

     apacheconf-cd 
            to apacheconf-dir

     apacheconf-makepatch 
            create patch from diff of original httpd.conf and current

     apacheconf-pristine-
            return code 0 for unmodified httpd.conf 
            
     apacheconf-applypatch
            apply the patch against the httpd.conf will only do so to a pristine httpd.conf
   
     apacheconf-diff
           diff to stdout 
        
     apacheconf-envvars-add <dir>
           add a directory to the *LD_LIBRARY_PATH felt by apachectl , directories are only added
           if not present already 
              
     apacheconf-original---
           copy the original httpd.conf to httpd.conf.original ... is invoked by the 
            apachebuild  NOT FOR USER USAGE 




     apacheconf-sqlite
           add sqlite lib dir to the apachectl envvars


   
EOU



}

apacheconf-env(){
  apache-
}


apacheconf-dir(){
  echo $APACHE_HOME/conf
}

apacheconf-cd(){
  cd $(apacheconf-dir)
}

apacheconf-path(){
  echo $(apacheconf-dir)/httpd.conf
}

apacheconf-bindir(){
  echo $APACHE_HOME/bin
}


apacheconf-patchpath(){
  echo  $ENV_HOME/apache/$APACHE_NAME.conf.patch
}




apacheconf-envvars-path(){
  echo $(apacheconf-bindir)/envvars
}

apacheconf-envvars-match-(){
   local add=$1
   local path=$(apacheconf-envvars-path)
   grep -qs $add $path 
}

apacheconf-envvars-add(){
   local add=$1
   local path=$(apacheconf-envvars-path)
   local msg="=== $FUNCNAME :"
   apacheconf-envvars-match- $add && echo $msg already has $add  present in envvars || apacheconf-envvars-add- $add
   grep _LIBRARY_PATH $path
}

apacheconf-envvars-add-(){
   local add=$1
   local msg="=== $FUNCNAME :"
   local path=$(apacheconf-envvars-path)
   echo $msg adding $add to envvars $path
   $ASUDO perl -pi -e "s,^(\S*_LIBRARY_PATH=\")(\S*)(\:\\$\S*_LIBRARY_PATH\")$,\$1$add:\$2\$3," $path
}



apacheconf-sqlite(){

   local msg="=== $FUNCNAME :"
   echo $msg THIS FUNCATIONALITY HAS BEEN MOVED TO  sqlite-again  
   sleep 100000000000000  && echo $msg SLEEPING UNTIL YOU FIX THIS 

}

apacheconf-original---(){
  local msg="=== $FUNCNAME :"
  local conf=$(apacheconf-path)
  [ ! -f $conf ] && echo $msg ERROR no conf $conf && sleep 10000000000
  $ASUDO cp $conf $conf.original  

}


apacheconf-makepatch(){

  local iwd=$PWD
  local msg="=== $FUNCNAME :"
  cd $APACHE_HOME
  
  local conf=$(apacheconf-path)
  conf=${conf/$APACHE_HOME\//}
  local orig=$conf.original

  local patch=$(apacheconf-patchpath)  
  echo $msg creating patch $patch REMEMBER TO svn add and ci FOR SAFE KEEPING
  diff -Naur $orig $conf  > $patch

  cd $iwd
}


apacheconf-pristine-(){

  local iwd=$PWD 
  local msg="=== $FUNCNAME :"
  local conf=$(apacheconf-path)
  
  cd $APACHE_HOME
  conf=${conf/$APACHE_HOME\//}
  local orig=$conf.original

  [ ! -f $conf ] && echo $msg no conf $conf && sleep 100000000
  [ ! -f $orig ] && echo $msg no orig $orig && sleep 100000000

  diff -q $orig $conf > /dev/null
  local ret=$?
  cd $iwd
  return $ret
}


apacheconf-diff(){

   local iwd=$PWD 
   
   local conf=$(apacheconf-path)
   conf=${conf/$APACHE_HOME\//}
   local orig=$conf.original
   
   cd $APACHE_HOME
   diff -Naur $orig $conf 
   local ret=$?
   cd $iwd
   return $ret
}


apacheconf-applypatch(){
   
  local iwd=$PWD 
  local msg="=== $FUNCNAME :" 
  local patch=$(apache-conf-patchpath)
  [ ! -f $patch ] && echo $msg there is no patch $patch skipping && return 1 
  
  cd $APACHE_HOME
  
  ! apache-conf-pristine- && echo $msg cannot apply patch as the conf is not pristine && return 1  
  
  echo $msg applying patch to pristine conf
  patch -p0 < $patch
    
  cd $iwd  
}


apacheconf-edit(){
  vi $(apacheconf-path)
}



