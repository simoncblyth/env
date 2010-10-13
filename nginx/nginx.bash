# === func-gen- : nginx/nginx.bash fgp nginx/nginx.bash fgn nginx
nginx-src(){      echo nginx/nginx.bash ; }
nginx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nginx-src)} ; }
nginx-vi(){       vi $(nginx-source) ; }
nginx-env(){      
   elocal- 
   pkgr- 
   env-append $(nginx-sbin) 
}

nginx-usage(){
  cat << EOU
     nginx-src : $(nginx-src)

     http://wiki.nginx.org/Main
     http://wiki.nginx.org/NginxXSendfile
     http://www.bitbucket.org/chris1610/satchmo/src/tip/satchmo/apps/satchmo_store/shop/views/download.py

     http://wiki.nginx.org/NginxCommandLine
        -s stop/quit/reopen/reload. (version >= 0.7.53)

     Putting nginx under supervisord control 
        http://www.vps.net/forum/public-forums/tutorials-and-how-tos/1102-how-to-spawn-php-with-supervisord-for-nginx-on-debian



    == PASSWORD PROTECTED DIR ==

    nginx-users  : $(nginx-users)
    nginx-adduser <username>
          you will be prompted for 
               * password 
               * salt (enter 2 chars at random)
   
          The protected location needs the following directives :
               auth_basic "realm-name-given-to-challengers" ;
               auth_basic_user_file  users.txt ;


    == redhat : installs from EPEL ==

     sudo yum --enablerepo=epel install nginx
         
       C      0.6.39-4.el4 
       N      0.6.39-4.el5 

      rpm -ql nginx  

          comes with a perl module interface to the nginx HTTP server API
             http://sysoev.ru/nginx/docs/http/ngx_http_perl_module.html





EOU
}


nginx-sv-(){ cat << EOX
# $FUNCNAME  ... requires  "daemon off;"  in nginx-edit 
[program:nginx]
command=$(which nginx)
process_name=%(program_name)s
autostart=true
autorestart=true

redirect_stderr=true
stdout_logfile=$(nginx-log)
stdout_logfile_maxbytes=5MB
stdout_logfile_backups=10


EOX
}
nginx-sv(){
  sv-
  $FUNCNAME- | sv-plus nginx.ini
}

nginx-user(){
  pkgr-
  case $(pkgr-cmd) in 
     yum) echo nginx ;;
       *) echo nobody ;;
  esac 
}

nginx-group(){
   echo $(nginx-user)
}

nginx-chown(){
  local msg="=== $FUNCNAME :"
  local cmd="sudo chown $(nginx-user):$(nginx-group) $* "
  echo $msg $cmd
  eval $cmd 
}



nginx-name(){ echo nginx-0.7.61 ; }
nginx-url(){  echo http://sysoev.ru/nginx/$(nginx-name).tar.gz ; }
nginx-dir(){  echo $(local-base)/env/nginx/$(nginx-name) ; }
nginx-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(nginx-dir)) && mkdir -p $dir
   cd $dir
   local ans
   read -p "$msg use your systems packager for simplicity ... " ans
   [ ! -d "$(nginx-name)" ] && curl -L -O $(nginx-url) && tar zxvf $(nginx-name).tar.gz  
}
nginx-build(){
   cd $(nginx-dir)

   ## using default prefix of /usr/local/nginx
   ./configure
   make
   sudo make install
}


nginx-prefix(){
  case $(pkgr-cmd) in 
    port) echo /opt/local ;; 
    ipkg) echo /opt ;;
     src) echo /usr/local/nginx     ;;   ## change this when revisit nginx on C
     yum) echo -n     ;;  
  esac
}

nginx-eprefix(){
  echo $(nginx-prefix)/usr ;
}


nginx-epel-install(){
  [ ! "$(pkgr-cmd)" == "yum" ] && return  
  if [ "$(rpm -ql nginx)" == "package nginx is not installed" ] ; then
      sudo yum --enablerepo=epel install nginx
  fi
}

nginx-epel-fixconf(){
  sudo perl -pi -e 's,/var/log/nginx/logs/access.log,/var/log/nginx/access.log, ' $(nginx-conf)
}

nginx-diff(){  sudo diff $(nginx-conf).default $(nginx-conf) ; } 


nginx-sbin(){    echo $(nginx-prefix $*)/sbin ; }
nginx-confd(){    echo $(nginx-prefix $*)/etc/nginx ; }
nginx-conf(){    echo $(nginx-confd $*)/nginx.conf ; }

 ## Since version 0.6.7 the filename path is relative to directory of nginx configuration file nginx.conf, but not to nginx prefix 
nginx-users(){   echo $(nginx-prefix)/etc/nginx/users.txt ; }
nginx-pidpath(){ 
  case $(pkgr-cmd) in 
    port) echo $(nginx-prefix)/var/run/nginx/nginx.pid ;;
    ipkg) echo $(nginx-prefix)/var/nginx/run/nginx.pid ;;
     yum) echo $(nginx-prefix)/var/run/nginx.pid ;;
  esac
}    
    
    
nginx-pid(){     cat $(nginx-pidpath) 2>/dev/null ; }
nginx-stop(){    sudo kill -QUIT $(nginx-pid) ; }
nginx-start(){   
   local msg="=== $FUNCNAME :"
   local pid=$(nginx-pid)
   [ -n "$pid" ] && echo $msg looks like nginx is running already with pid $pid from pidfile $(nginx-pidpath) ... stop it first && return 0
   sudo nginx ; 
}

nginx-strings(){
   strings $(which nginx) | grep nginx.conf
}
nginx-info(){  cat << EOI
    nginx-pid : $(nginx-pid) 
EOI
}


nginx-htdocs(){ echo $(nginx-eprefix)/share/nginx/html ; }
nginx-logd(){   
   case $(pkgr-cmd) in 
     port)  echo $(pkgr-logd)/nginx  ;;
     ipkg)  echo $(nginx-prefix)/var/nginx/log ;;
      yum)  echo $(nginx-prefix)/var/log/nginx ;;
   esac  
}
nginx-cd(){     cd $(nginx-logd) ; }
nginx-log(){    echo $(nginx-logd)/sv.log ; }
nginx-elog(){   echo $(nginx-logd)/error.log ; }
nginx-alog(){   echo $(nginx-logd)/access.log ; }
nginx-tail(){  sudo tail -f $(nginx-log) ; }
nginx-etail(){  sudo tail -f $(nginx-elog) ; }
nginx-atail(){  sudo tail -f $(nginx-alog) ; }



nginx-adduser-(){
   local pass=$1
   local salt=$2
   cat << EOC
perl -le "print crypt('$pass', '$salt');" 
EOC
}

nginx-adduser(){
  local msg="=== $FUNCNAME :"
  local user=$1
  local pass
  read -p "$msg enter password for user \"$user\" :" pass
  read -p "$msg repeat password :" pass2
  [ "$pass" != "$pass2" ] && echo $msg passwords dont match ... && return 1
  local salt
  read -p "$msg enter salt :" salt
  local comm
  read -p "$msg enter comment :" comm

  local cmd=$(nginx-adduser- $pass $salt)
  echo $msg $cmd
  local hash=$(eval $cmd) 

  echo $msg appending new user entry "$user:$hash:$comm" to users file  $(nginx-users) 
  sudo bash -c "echo \"$user:$hash:$comm\" >> $(nginx-users) "

}





nginx-edit(){  sudo vim $(nginx-conf) ; }
nginx-ps(){ ps aux | grep nginx ; }

# this is only for newer nginx
#nginx-stop(){
#    sudo nginx -s stop
#}

nginx-check(){
   sudo $(nginx-sbin)/nginx -c $(nginx-conf) -t 
}





nginx-rproxy-(){  cat << EOC
worker_processes 1 ;
events { worker_connections  1024; }
http {
    server {
        listen       80;
        server_name localhost ;
        location / {
              proxy_pass     http://picasaweb.google.com ;
              proxy_redirect http://picasaweb.google.com/ /;
              proxy_set_header  X-Real-IP  \$remote_addr;
        }
    }
}
EOC
}

nginx-rproxy-rconf(){  echo conf/local/rproxy.conf ; }
nginx-rproxy-conf(){   echo $(nginx-prefix)/$(nginx-rproxy-rconf) ; }
nginx-rproxy(){
   local msg="=== $FUNCNAME :"
   local conf=$(nginx-rproxy-conf)  
   local tmp=/tmp/env/$FUNCNAME/$(basename $conf) && mkdir -p $(dirname $tmp)
   $FUNCNAME- > $tmp
   echo $msg wrote $tmp ... replace $conf ?
   cat $tmp
   local cmd="sudo mkdir -p $(dirname $conf) && sudo cp $tmp $conf "
   echo $msg $cmd
   eval $cmd
}

nginx-rproxy-start(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo nginx -c $(nginx-rproxy-rconf) "
   echo $cmd
   eval $cmd 
}



nginx-ln(){

   local iwd=$PWD
   local dir=$1
   [ -z "$dir" ] && echo enter the directory to link into nginx-htdocs : $(nginx-htdocs) && return 
   local name=${2:-$(basename $dir)}

   cd $(nginx-htdocs)
   local cmd="sudo ln -s $dir $name "
   echo $cmd
   eval $cmd
   cd $iwd


}

nginx-ls(){
 ls -l  `nginx-htdocs`/
}


