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
   local mode=$(apache-mode $*)
   local oldbin=$(apache-bin)     ## tis old because APACHE_MODE has not been chnaged yet
   export APACHE_MODE=$mode
   export APACHE_HOME=$(apache-home)   ## should be able to get rid of this envvar ?
   local bin=$(apache-bin)  

   if [ "$oldbin" != "$bin" ]; then 
      env-remove $oldbin
      env-prepend $bin
   fi
}

apache-mode(){ 
   local arg=$1
   [ -z "$arg" ] && echo ${APACHE_MODE:-source} && return 0
   if [ "${arg:0:6}" == "system" ]; then
       echo $arg  
   else
       echo source 
   fi  
}

apache-name(){
   local tag=${1:-$NODE_TAG}
   case $tag in 
      H) echo httpd-2.0.59 ;;
      *) echo httpd-2.0.63 ;;
   esac
}

apache-target(){ echo http://cms01.phys.ntu.edu.tw ;  }   ## WHO USES THIS ?


##
##   apache user / group 
##
   
apache-user(){ perl -n -e 's,^User\s*(\S*),$1, && print ' $(apache-conf) ;  } ## local only 
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

apache-sudouser(){ [ -n "$SUDO" ] && echo $SUDO -u $(apache-user) || echo "" ; }

apache-chown(){
  local msg="=== $FUNCNAME :"
  local path=$1
  shift
  local opts=$*
  local cmd="sudo chown $opts $(apache-user):$(apache-group) $path "
  echo $msg $cmd
  eval $cmd
}

apache-chcon(){ sudo chcon -R -h -t httpd_sys_content_t $1 ;  }


##
##   characterization of many apaches
##
apache-info(){
   cat << EOI
     APACHE_MODE       : $APACHE_MODE
     APACHE_HOME       : $APACHE_HOME


     which apachectl   : $(which apachectl 2> /dev/null)


     apache-mode       : $(apache-mode)
     apache-sysflavor  : $(apache-sysflavor)

            change the mode/flavor with the precursor, eg 

                  apache- systemapple
                  apache- systemport
                  apache- systemyum
                  apache- source

     apache-home       : $(apache-home)
     apache-bin        : $(apache-bin)
     apache-confdir    : $(apache-confdir)
     apache-confd      : $(apache-confd)
     apache-htdocs     : $(apache-htdocs)
     apache-modulesdir : $(apache-modulesdir)
     apache-logdir     : $(apache-logdir)

EOI
}


apache-sudo(){      apache-issystem- && echo sudo || echo -n  ; }
apache-issystem-(){ [ "${APACHE_MODE:0:6}" == "system" ] && return 0 || return 1  ; }
apache-sysflavor-default(){
    case $(uname) in 
      Darwin) echo port ;;
       Linux) echo yum ;;
    esac
}
apache-sysflavor(){ 
    [ "${APACHE_MODE:0:6}" != "system" ] && echo -n && return 0
    local flavor=${APACHE_MODE:6}
    case $flavor in 
       port|yum|apple) echo $flavor ;;
                    *) echo $(apache-sysflavor-default) ;;
    esac
}



apache-check-(){
   local msg="=== $FUNCNAME :"
   local rc=0
   apache-info
   local fns="home bin confdir confd htdocs modulesdir logdir"
   local fn
   for fn in $fns ; do
       local func=apache-$fn-check-
       ! $func  && echo $msg FAILED   $func $(apache-$fn) && rc=${#check}
          $func && echo $msg SUCEEDED $func $(apache-$fn) 
   done
   return $rc
}

##
##

apache-home-check-(){  [ -d "$(apache-home)" ] && return 0 || return 1 ; }
apache-home(){         local tag=${1:-$NODE_TAG} ; apache-issystem- && apache-home-system $tag  || apache-home-source $tag ; }
apache-home-source(){
  local tag=${1:-$NODE_TAG}
  case $tag in 
     H) echo $(local-base $tag)/apache2/$(apache-name $tag) ;;
     *) echo $(local-system-base $tag)/apache/$(apache-name $tag) ;;
  esac
}
apache-home-system(){
      local tag=${1:-$NODE_TAG}
      local flavor=$(apache-sysflavor)
      local label=$tag_$flavor
      case $flavor in 
             apple) echo /Library/WebServer  ;;
              port) echo /opt/local/apache2 ;; 
               yum) echo /var/www  ;;
                 *) echo failed-$FUNCNAME ;;
      esac
}
## 
##
apache-bin-check-(){  [ -f "$(apache-bin)/apachectl" ] && return 0 || return 1 ; }
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
      case $flavor in 
             apple) echo /usr/sbin  ;;
              port) echo /opt/local/apache2/bin ;; 
               yum) echo /usr/sbin  ;;
                 *) echo failed-$FUNCNAME ;;
      esac
}
apache-envvars(){ echo $(apache-bin $*)/envvars ; }

## 
##
apache-confdir-check-(){  [ -f "$(apache-confdir)/httpd.conf" ] && return 0 || return 1 ; }
apache-confdir(){ local tag=${1:-$NODE_TAG} ; apache-issystem- && apache-confdir-system $tag  || apache-confdir-source $tag ; }
apache-confdir-source(){
  local tag=${1:-$NODE_TAG} 
  case $tag in
        H) echo $(apache-home $tag)/etc/apache2 ;;
        *) echo $(apache-home $tag)/conf ;;
  esac
}
apache-confdir-system(){
  local tag=${1:-$NODE_TAG} 
  local flavor=$(apache-sysflavor)
  case $flavor in
        apple) echo /private/etc/apache2 ;;
         port) echo /opt/local/apache2/conf ;; 
          yum) echo /etc/httpd/conf ;;
            *) echo failed-$FUNCNAME ;;
  esac
}

apache-fragmentpath(){ echo $(apache-confdir)/${1:-fragment}.conf ; }
apache-conf(){         echo $(apache-confdir $*)/httpd.conf ; }   
apache-edit(){         $SUDO vi $(apache-conf) ; }

apache-addline(){
  local msg="=== $FUNCNAME :"
  local line=$1
  local conf=$(apache-conf)
  local sudouser=$(apache-sudouser)
  grep -q "$line" $conf && echo $msg line \"$line\" already present in $conf  || $sudouser echo "$line" >> $conf  
}

apache-confd-check-(){  [ -d "$(apache-confd)" ] && return 0 || return 1 ; }
apache-confd(){
   ## used by svnsetup-sysapache
   local confdir=$(apache-confdir $*)
   local confd=$(dirname $confdir)/conf.d 
   [ -d "$confd" ] && echo $confd || echo /tmp
}

##
##

apache-htdocs-check-(){ [ -d "$(apache-htdocs)" ] && return 0 || return 1 ; }
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
        port) echo /opt/local/apache2/htdocs  ;;
         yum) echo /var/www/html ;;
           *) echo failed-$FUNCNAME ;;
  esac  
}

apache-docroot(){      apache-htdocs $* ; }
apache-downloadsdir(){ local tag=${1:-$NODE_TAG} ; echo $(apache-htdocs $tag)/downloads ; }
apache-docroot-local(){ grep DocumentRoot $(apache-conf) | perl -n -e 'm,^DocumentRoot\s*\"(\S*)\", && print $1 ' ; }

##
##

apache-modulesdir-check-(){ [ -d "$(apache-modulesdir)" ] && return 0 || return 1 ; }
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
  local flavor=$(apache-sysflavor)
  case $flavor in 
     apple) echo /usr/libexec/apache2 ;;
      port) echo /opt/local/apache2/modules ;;
       yum) echo /usr/lib/httpd/modules ;;
         *) echo failed-$FUNCNAME ;;
  esac    
}

apache-ls(){   ls -alst $(apache-modulesdir) ; }

##
##
apache-logdir-check-(){ [ -d "$(apache-logdir)" ] && return 0 || return 1 ; }
apache-logdir(){ local tag=${1:-$NODE_TAG} ; apache-issystem- && apache-logdir-system $tag  || apache-logdir-source $tag ; }
apache-logdir-source(){
  local tag=${1:-$NODE_TAG}
  case $tag in 
     H) echo $(apache-home $tag)/var/apache2/log ;;
     *) echo $(apache-home $tag)/logs ;;
  esac    
}
apache-logdir-system(){
  local tag=${1:-$NODE_TAG}
  local flavor=$(apache-sysflavor)
  case $flavor in 
     apple) echo /var/log/apache2 ;;
      port) echo /opt/local/apache2/logs ;;
       yum) echo /var/log/httpd ;;
         *) echo failed-$FUNCNAME ;;
  esac    
}

apache-etail(){ $(apache-sudo) tail -f $(apache-logdir)/error_log ; }   
apache-evi(){   $(apache-sudo)      vi $(apache-logdir)/error_log ; }   
apache-atail(){ $(apache-sudo) tail -f $(apache-logdir)/access_log ; }   
apache-avi(){   $(apache-sudo)      vi $(apache-logdir)/access_log ; }   
apache-logs(){ cd $(apache-logdir) ;  ls -l ; }


## publish a dir via a link  in htdocs
 
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


## nefarious 

