tracpreq-src(){   echo trac/tracpreq.bash ; }
tracpreq-source(){ echo ${BASH_SOURCE:-$(env-home)/$(tracpreq-src)} ; }
tracpreq-vi(){     vi $(tracpreq-source) ; }
tracpreq-usage(){

  cat << EOU
  
      tracpreq-again 
            build again the prerequisites for a trac server ...
            CAUTION existing installations ARE WIPED 
            
            ensure any build and install directory customisations 
            are safety tucked away in patches before doing this   
 

     This is the first use of the env-log machinery ...
     twould be muchmore convenient for env-log calls not
     to have to pass their top level "context" , FUNCNAME in the below
     in order that the env-log calls could be reused 

     Could do that via a symbolic link that on doing an initlog points to 
     the context ...
        /tmp/env/logs/
            lasttopfunc -> tracpreq-again
 
     Then subsequent log calls can write via 2 symbolic links to  
       /tmp/env/logs/lasttopfunc/last.log
     this allows env-log calls to only need to report their immediate context
     ... so can be used thru multi levels of calls 


 
EOU

}

tracpreq-env(){
   echo -n
}

tracpreq-mode(){ echo ${TRACPREQ_MODE:-$(tracpreq-mode-default $*)} ; }
tracpreq-mode-default(){
   case ${1:-$NODE_TAG} in
  ZZ|C|Y1|Y2) echo system ;;
           *) echo source ;;
   esac
}


tracpreq-again(){ tracpreq-again-$(tracpreq-mode) ; }
tracpreq-again-system(){
   [ "$(tracpreq-mode)" != "system" ] && echo $msg ABORT this is for tracpreq-mode:system  only && return 1

   tracpreq-py

}


tracpreq-py(){

   setuptools-
   setuptools-get    # downloads and runs the installer script

   configobj-          
   configobj-build   # does get(from SVN trunk)/build-/install

}


tracpreq-again-source(){

   local msg="=== $FUNCNAME :"
   [ "$(tracpreq-mode)" != "source" ] && echo $msg ABORT this is for tracpreq-mode:source only && return 1

   local ans
   read -p "$msg ENTER YES TO PROCEED" ans
   [ "$ans" != "YES" ] && echo $msg skipping && return 1

   log-
   log-init $FUNCNAME   

   python- 
   pythonbuild-       
   pythonbuild-again   | log-- $FUNCNAME pythonbuild-again
  

  # THIS NEEDS svn WHICH DONT HAVE YET  
  # configobj-          
  # configobj-get       | log-- $FUNCNAME configobj-get 
   
   
   swig-               
   swigbuild-           
   swigbuild-again     | log-- $FUNCNAME swigbuild-again
   
   apache-
   apache-again        | log-- $FUNCNAME apache-again
   
   svn-
   svnbuild-
   svnbuild-again      | log-- $FUNCNAME svnbuild-again

   sqlite-
   sqlite-again        | log-- $FUNCNAME sqlite-again


   tracpreq-py


}


tracpreq-system(){
  [ "$(which yum)" != "" ] && $FUNCNAME-yum $* 
}

tracpreq-system-yum-(){ cat << EOS
python
mod_dav_svn
subversion
mod_python
swig
sqlite
httpd
EOS
}

tracpreq-system-yum(){
  local cmd=${1:-info}
  local msg="=== $FUNCNAME :"
  echo $msg WARNING THIS HAS NEVER BEEN USED 
  local line
  $FUNCNAME- | while read line ; do
     sudo yum $cmd $line
  done
}




