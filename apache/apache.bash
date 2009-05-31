apache-src(){    echo apache/apache.bash ; }
apache-source(){ echo $ENV_HOME/$(apache-src) ; }
apache-vi(){     vi $(apache-source) ; }

apachebuild-(){ . $ENV_HOME/apache/apachebuild/apachebuild.bash && apachebuild-env $* ; } 
apacheconf-(){  . $ENV_HOME/apache/apacheconf/apacheconf.bash   && apacheconf-env $* ; } 
apachepriv-(){  . $ENV_HOME/apache/apacheconf/apachepriv.bash   && apachepriv-env $* ; } 
modpython-(){   . $ENV_HOME/apache/apachebuild/modpython.bash   && modpython-env $* ; }
mpinfo-(){      . $ENV_HOME/apache/apachebuild/mpinfo.bash      && mpinfo-env $* ; } 
iptables-(){    . $ENV_HOME/apache/apacheconf/iptables.bash     && iptables-env $* ; }



apache-usage(){

   cat << EOU
 
     apache-src      : $(apache-src)
     apache-vi


       which apachectl : $(which apachectl) 

      apache-ls
                ls dso modules
      apache-logs 
                ls logs 
   
      NODE_TAG           : $NODE_TAG
      
      NB where it is likely to be useful to grab the 
      value for another node ... its best to implement 
      in a fully functional manner.. allowing simple access
      like 
          remval=$(NODE_TAG=R apache-confdir)
      
      rather than fooling with apache- to set the exports 
      
      
      apache-name        : $(apache-name)
      apache-user        : $(apache-user)
      apache-home        : $(apache-home)
      apache-mode        : $(apache-mode)
              system or source

      apache-envvars     : $(apache-envvars)
      apache-target      : $(apache-target)
      apache-confdir     : $(apache-confdir)
      apache-confd       : $(apache-confd)
      apache-fragmentpath demo :  $(apache-fragmentpath demo)    
      apache-modulesdir  : $(apache-modulesdir)
      apache-bin         : $(apache-bin)
      apache-htdocs      : $(apache-htdocs)
      apache-logdir      : $(apache-logdir)
      apache-downloadsdir  : $(apache-downloadsdir)
      
      apache-sudouser    : $(apache-sudouser)
      
      apache-again
           wipes and builds both apache and modpython
           CAUTION:  this wipes installations and rebuilds from the tarball
   
   
      Precursors ...
      
         apachebuild-
         apacheconf-
         modpython-
         mpinfo-
         iptables-
         

     When attempting use of system apache, will need to
         sudo yum install httpd
         sudo yum install httpd-devel   ## for apxs
     
   
   
EOU


}

apache-again(){
    
    apachebuild-
    apachebuild-again
    
    modpython-
    modpython-again

}




apache-env(){
   elocal-
   local mode=${1:-$(apache-mode)}
   if [ "$mode" == "system" ]; then
      [ -n "$APACHE_HOME" ] && env-remove $APACHE_HOME/bin
      env-prepend /usr/sbin
      export APACHE_MODE=system
   else
      env-remove /usr/sbin
      export APACHE_MODE=source
      export APACHE_NAME=$(apache-name)
      export APACHE_HOME=$(apache-home)   
      [ -d $APACHE_HOME ] && env-prepend $APACHE_HOME/bin
  fi
}


apache-mode(){ 
   echo ${APACHE_MODE:-source} ; 
   #env-inpath- apache && echo source || echo system
}

apache-home(){
   local tag=${1:-$NODE_TAG}
   local mode=$(apache-mode)
   if [ "$mode" == "system" ]; then
      case $tag in 
        *) echo /etc/httpd ;; 
      esac
   else
      case $tag in 
        H) echo $(local-base $tag)/apache2/$(apache-name $tag) ;;
        *) echo $(local-system-base $tag)/apache/$(apache-name $tag) ;;
      esac
   fi
}

apache-name(){

 #
 # httpd-2.0.59  known to work with svn bindings for Trac usage
 # httpd-2.0.61  nearest version on the mirror    
 # httpd-2.0.63  nearest version on the mirror    
 #
   local tag=${1:-$NODE_TAG}
   case $tag in 
      H) echo httpd-2.0.59 ;;
      *) echo httpd-2.0.63 ;;
   esac
}

apache-target(){
  echo http://cms01.phys.ntu.edu.tw
}

apache-envvars(){
  local tag=${1:-$NODE_TAG}
  case $tag in 
   H) echo $(apache-home $tag)/sbin/envvars ;;
   *) echo $(apache-home $tag)/bin/envvars ;;
  esac 
}

apache-confd(){
   ## used by svnsetup-sysapache
   case ${1:-$NODE_TAG} in 
      N) echo /etc/httpd/conf.d ;;
      *) echo /etc/httpd/conf.d ;;
   esac
}

apache-user(){
   ## not convenient to do this remotely ...
   perl -n -e 's,^User\s*(\S*),$1, && print ' $(apache-conf)
}

apache-user-deprecated(){
   case ${1:-$NODE_TAG} in 
     G) echo www ;;
  C|C2) echo nobody ;;
  P|G1) echo dayabaysoft ;;
     N) echo apache ;;
     *) echo apache ;;
   esac
}

apache-group(){
   local tag=${1:-$NODE_TAG}
   case $tag in 
  P|G1) echo dayabay ;;
     *) echo $(apache-user $tag) ;;
   esac
}

apache-chown(){
  local msg="=== $FUNCNAME :"
  local path=$1
  shift
  local opts=$*
  local cmd="sudo chown $opts $(apache-user):$(apache-group) $path "
  echo $msg $cmd
  eval $cmd

}

apache-se-content(){
   local path=$1
   sudo chcon -R -h -t httpd_sys_content_t $path
}

apache-confdir(){
  local tag=${1:-$NODE_TAG} 
  case $tag in
        G) echo /private/etc/apache2 ;;
        H) echo $(apache-home $tag)/etc/apache2 ;;
        C) echo $(apache-home $tag)/conf ;;
        *) echo $(apache-home $tag)/conf ;;
  esac
}


apache-issystem-(){ [ "${APACHE_MODE:0:6}" == "system" ] && return 0 || return 1  ; }
apache-sysflavor-default(){
    case $(uname) in 
      Darwin) echo port ;;
       Linux) echo yum ;;
    esac
}
apache-sysflavor(){ 
    local flavor=${APACHE_MODE:6}
    [ -n "$flavor" ] && echo $flavor || echo $(apache-sysflavor-default) 
}


## the dir with apachectl 

apache-bin(){    local tag=${1:-$NODE_TAG} ; apache-issystem- && apache-bin-system $tag  || apache-bin-source $tag ; }
apache-bin-source(){
  local tag=${1:-$NODE_TAG}
  case $tag in 
     H) echo  $(apache-home $tag)/sbin  ;;
     *) echo  $(apache-home $tag)/bin  ;;
  esac 
}
apache-bin-system(){
      local tag=${1:-$NODE_TAG}
      local flavor=$(apache-sysflavor)
      local label=$tag_$flavor
      case $flavor in 
             apple) echo /usr/sbin  ;;
              port) echo /opt/local/apache2/bin ;; 
               yum) echo /usr/sbin  ;;
      esac
}

## 

apache-htdocs(){ local tag=${1:-$NODE_TAG} ; apache-issystem- && apache-htdocs-system $tag  || apache-htdocs-source $tag ; }
apache-htdocs-source(){
  local tag=${1:-$NODE_TAG}
  case $tag in 
    H) echo $(apache-home $tag)/share/apache2/htdocs ;;
    *) echo $(apache-home $tag)/htdocs  ;;
  esac  
}
apache-htdocs-system(){
  local tag=${1:-$NODE_TAG}
  local flavor=$(apache-sysflavor)
  case $flavor in 
       apple) echo /Library/WebServer/Documents ;;
        port) echo $FUNCNAME  ;;
         yum) echo /var/www/html ;;
  esac  
}

##

apache-modulesdir(){ local tag=${1:-$NODE_TAG} ; apache-issystem- && apache-modulesdir-system $tag  || apache-modulesdir-source $tag ; }
apache-modulesdir-source(){
  local tag=${1:-$NODE_TAG}
  case $tag in 
     H) echo $(apache-home $tag)/libexec ;;
     *) echo $(apache-home $tag)/modules ;;
  esac    
}
apache-modulesdir-system(){
  local tag=${1:-$NODE_TAG}
  case $tag in 
     G) echo /usr/libexec/apache2 ;;
  esac    
}





## nefarious 

apache-downloadsdir(){
  local tag=${1:-$NODE_TAG}
  case $tag in 
    *) echo $(apache-htdocs $tag)/downloads  ;;
  esac  
}


apache-docroot(){ apache-htdocs $* ; }
apache-docroot-(){ grep DocumentRoot $(apache-conf) | perl -n -e 'm,^DocumentRoot\s*\"(\S*)\", && print $1 ' ; }



apache-fragmentpath(){
   echo $(apache-confdir)/${1:-fragment}.conf
}

apache-logdir(){
   case ${1:-$NODE_TAG} in 
      N) echo /var/log/httpd ;;
      G) echo /var/log/apache2 ;;
      H) echo $APACHE_HOME/var/apache2/log ;;
      *) echo $APACHE_HOME/logs ;;
   esac   
}   
 
 
apache-publish-logdir(){
   local msg="=== $FUNCNAME :" 
   local dir=$1
   local name=${2:-$(basename $dir)}
   [ ! -d $dir ] && echo $msg ABORT no such dir $dir && return 1
   local iwd=$PWD
   cd `apache-htdocs`
   [ ! -d logs ] && mkdir -p logs
   cd logs
   echo $msg creating link in $PWD, from $dir to $name 
   ln -sf $dir $name
   cd $iwd
}

apache-conf(){
   local tag=${1:-$NODE_TAG}
   echo $(apache-confdir $tag)/httpd.conf
}   
     
apache-edit(){
   $SUDO vi $(apache-conf)
}
         
apache-sudouser(){
  local user=$(apache-user)
  [ -n "$SUDO" ] && echo $SUDO -u $user || echo ""
}
     
apache-addline(){

  local msg="=== $FUNCNAME :"
  local line=$1
  local conf=$(apache-conf)
  local sudouser=$(apache-sudouser)

  grep -q "$line" $conf && echo $msg line \"$line\" already present in $conf  || $sudouser echo "$line" >> $conf  

}
   
apache-sudo(){
   case $(apache-mode) in 
     source) echo -n ;; 
          *) echo sudo ;;
   esac
}


apache-etail(){ $(apache-sudo) tail -f $(apache-logdir)/error_log ; }   
apache-evi(){   $(apache-sudo) vi $(apache-logdir)/error_log ; }   
apache-atail(){ $(apache-sudo) tail -f $(apache-logdir)/access_log ; }   
apache-avi(){   $(apache-sudo) vi $(apache-logdir)/access_log ; }   
   
apache-logs(){
  cd $(apache-logdir)
  ls -l 
}

apache-ls(){
   ls -alst $(apache-modulesdir)
}
