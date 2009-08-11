# === func-gen- : lighttpd/lighttpd.bash fgp lighttpd/lighttpd.bash fgn lighttpd
lighttpd-src(){      echo lighttpd/lighttpd.bash ; }
lighttpd-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lighttpd-src)} ; }
lighttpd-vi(){       vi $(lighttpd-source) ; }
lighttpd-env(){      pkgr- ; }
lighttpd-usage(){
  cat << EOU
     lighttpd-src : $(lighttpd-src)


    start from the port default :
       g4pb:lighttpd blyth$ sudo cp lighttpd.conf.default lighttpd.conf




     http://blog.lighttpd.net/articles/2006/07/18/reverse-proxying-mod_proxy_core
     http://redmine.lighttpd.net/projects/1/wiki/Docs:ModProxyCore

     lighttpd-rproxy  ... test a non-default config with :
          LIGHTTPD_CONF=$(lighttpd-rproxy-conf) lighttpd-check

      It appears that reverse proxying is no go in 1.4.22 that is coming from dag repo


EOU
}

lighttpd-check(){   sudo lighttpd -f $(lighttpd-conf) -p  ; }
lighttpd-base(){    echo $(pkgr-prefix)/etc/lighttpd ; }
lighttpd-initd(){   echo $(pkgr-prefix)/etc/init.d ; }
lighttpd-cd(){      cd $(lighttpd-base) ; }

#lighttpd-check(){   $(pkgr-sbin)/lighttpd -f $(lighttpd-conf) -p  ; }

lighttpd-conf(){    echo ${LIGHTTPD_CONF:-$(lighttpd-base)/lighttpd.conf} ; }
lighttpd-confd(){   echo $(lighttpd-base)/conf.d  ;  }
lighttpd-edit(){    sudo vim $(lighttpd-conf) $(lighttpd-confd)/*.conf ; }
lighttpd-ini(){     
  case $(pkgr-cmd) in 
    yum) echo /etc/rc.d/init.d/lighttpd ;;
   ipkg) echo /opt/etc/init.d/S80lighttpd ;; 
   port) echo /opt/local/etc/LaunchDaemons/org.macports.lighttpd/lighttpd.wrapper ;;
  esac 
}

lighttpd-init(){    sudo $(lighttpd-ini) $* ; }
lighttpd-start(){   lighttpd-init start ; }
lighttpd-stop(){    lighttpd-init stop ; }
lighttpd-restart(){ lighttpd-init restart ; }
lighttpd-ps(){     ps aux | grep lighttpd | grep -v grep ; }


lighttpd-htdocs(){ echo $(pkgr-wwwd)/lighttpd/htdocs ; }

## logging 

lighttpd-logd(){   echo $(pkgr-logd)/lighttpd  ; }
lighttpd-elog(){   echo $(lighttpd-logd)/error.log ; }
lighttpd-alog(){   echo $(lighttpd-logd)/access.log ; }
lighttpd-etail(){  sudo tail -f $(lighttpd-elog) ; } 
lighttpd-atail(){  sudo tail -f $(lighttpd-alog) ; } 
lighttpd-pidf(){   echo $(pkgr-rund)/lighttpd.pid ; }
lighttpd-pid(){    [ -f "$(lighttpd-pidf)" ] && cat $(lighttpd-pidf) ; }

lighttpd-port(){   echo 9000 ; }


lighttpd-config-(){ cat << EOC

## CAUTION THIS IS MANUALLY PROPAGATED INTO THE CONF

server.document-root        = "$(lighttpd-htdocs)/"

## this matches the pid path in the macports lighttpd.wrapper 
server.port                 = "$(lighttpd-port)"
server.pid-file             = "$(lighttpd-pidf)"
server.errorlog             = "$(lighttpd-elog)"
accesslog.filename          = "$(lighttpd-alog)"

EOC

  [ "$(uname)" == "Darwin" ] && ${FUNCNAME}osx-
}


lighttpd-config-osx-(){ cat << EOX

## set the event-handler (read the performance section in the manual)
server.event-handler = "freebsd-kqueue" # needed on OS X

EOX
}

lighttpd-config(){
   local msg="=== $FUNCNAME :"

   echo $msg manually merge the below with $(lighttpd-conf)
   $FUNCNAME- 

   local dirs="$(lighttpd-logd) $(lighttpd-rund) $(lighttpd-htdocs)"
   echo $msg prepare dirs $dirs 
   local dir
   for dir in $dirs ; do
     [ ! -d "$dir" ] && echo $msg create $dir && sudo mkdir -p $dir 
   done 
}





lighttpd-rproxy-conf(){ echo $(dirname $(lighttpd-conf))/rproxy.conf ; }
lighttpd-rproxy(){
    local msg="=== $FUNCAME :"
    local conf=$($FUNCNAME-conf)
    local tmp=/tmp/env/$FUNCNAME/$(basename $conf) && mkdir -p $(dirname $tmp)
    $FUNCNAME-  > $tmp
    cat $tmp
    echo $msg wrote $tmp ... replacing $conf 
    local cmd="sudo cp $tmp $conf "
    echo $msg $cmd
    eval $cmd

   echo $msg checking config :
   LIGHTTPD_CONF=$(lighttpd-rproxy-conf) lighttpd-check
}

lighttpd-rproxy-start(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo lighttpd -f $(lighttpd-rproxy-conf) "
   echo $msg $cmd
   eval $cmd
}


lighttpd-rproxy-port(){ echo 8000 ; }
lighttpd-rproxy-(){  
   local proxyme=${1:-/picasa}
   local backend=${2:-picasaweb.google.com}
   local port=${3:-$(lighttpd-rproxy-port)}
   cat << EOC

server.document-root        = "$(lighttpd-htdocs)/"
server.modules  += ( "mod_proxy_backend_http" )

## this matches the pid path in the macports lighttpd.wrapper 
server.port                 = $port
server.pid-file             = "$(lighttpd-pidf)"
server.errorlog             = "$(lighttpd-elog)"
accesslog.filename          = "$(lighttpd-alog)"


\$HTTP["url"] =~ "^$proxyme(/|$)" {
  proxy-core.balancer = "round-robin" 
  proxy-core.protocol = "http" 
  proxy-core.backends = ( "$backend" )
  proxy-core.rewrite-response = (
    "Location" => ( "^http://$backend/(.*)" => "http://127.0.0.1:$port$proxyme/\$1" ),
  )
  proxy-core.rewrite-request = (
    "_uri" => ( "^$proxyme/?(.*)" => "/\$1" ),
    "Host" => ( ".*" => "$backend" ),
 )
}

EOC

}



