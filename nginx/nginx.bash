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

NGINX
======

* http://wiki.nginx.org/Main
* http://wiki.nginx.org/NginxXSendfile

   * http://www.bitbucket.org/chris1610/satchmo/src/tip/satchmo/apps/satchmo_store/shop/views/download.py

Putting nginx under supervisord control 

* http://www.vps.net/forum/public-forums/tutorials-and-how-tos/1102-how-to-spawn-php-with-supervisord-for-nginx-on-debian

fastcgi
--------

* http://wiki.nginx.org/HttpFastcgiModule


Commandline
------------

* http://wiki.nginx.org/NginxCommandLine


INSTALLS
----------

N
~~

Controlled via supervisord::

    [blyth@belle7 ~]$ sv
    demo_listener                    RUNNING    pid 3030, uptime 90 days, 6:38:18
    demo_listener_raw                RUNNING    pid 3029, uptime 90 days, 6:38:18
    demo_logger                      RUNNING    pid 3047, uptime 90 days, 6:38:18
    dybslv                           RUNNING    pid 20321, uptime 0:43:56
    hgweb                            RUNNING    pid 3045, uptime 90 days, 6:38:18
    mysql                            RUNNING    pid 3046, uptime 90 days, 6:38:18
    nginx                            FATAL      Exited too quickly (process log may have details)
    N> stop nginx
    nginx: ERROR (not running)
    N> start nginx
    nginx: started
    N> tail -f nginx
    ==> Press Ctrl-C to exit <==
    2013/11/03 15:13:44 [emerg] 1596#0: open() "/etc/nginx/scgi_params" failed (2: No such file or directory) in /etc/nginx/nginx.conf:126
    2013/11/03 15:13:45 [emerg] 1827#0: open() "/etc/nginx/scgi_params" failed (2: No such file or directory) in /etc/nginx/nginx.conf:126
    2013/11/03 15:13:48 [emerg] 2249#0: open() "/etc/nginx/scgi_params" failed (2: No such file or directory) in /etc/nginx/nginx.conf:126
    2013/11/03 15:13:51 [emerg] 2611#0: open() "/etc/nginx/scgi_params" failed (2: No such file or directory) in /etc/nginx/nginx.conf:126



hfag.phys.ntu.edu.tw
~~~~~~~~~~~~~~~~~~~~~

Used as reverse proxy for SVN on C2, allowing NUU nodes that are routinely 
blocked from accessing node C2 apache (for SVN) to have access to the repositories. 
The NUU blocks are assumed to be due to C2s habit of transferring 
gigabytes per day of backup tarballs. 

*Not auto-started on reboot*, to start::

   nginx-
   nginx--sstart
 

HOW TO EXPOSE SOME STATIC HTML DOCS
--------------------------------------

For example the numpy documentation::

        cd `nginx-htdocs` 
        sudo ln -s /data/env/local/env/npy/numpy/doc/build/html np
        nginx-edit      ## create section similar to /logs   
        nginx-stop      
        
        ##  sv will auto restart with the new config
        visit http://cms01.phys.ntu.edu.tw/np/


FUNCTIONS
----------

nginx-users  : $(nginx-users)

nginx-adduser <username>

          you will be prompted for 

               * password 
               * salt (enter 2 chars at random)
  
Protecting directory
-----------------------

#. Create users. 
#. add config directives::

  auth_basic "realm-name-given-to-challengers" ;
  auth_basic_user_file  users.txt ;


INSTALLS
-----------

======== =================
 node      version
======== =================  
   C      0.6.39-4.el4 
   N      0.6.39-4.el5 
======== =================  

::

        -s stop/quit/reopen/reload. (version >= 0.7.53)


From EPEL::

     sudo yum --enablerepo=epel install nginx
     rpm -ql nginx  

comes with a perl module interface to the nginx HTTP server API

 * http://sysoev.ru/nginx/docs/http/ngx_http_perl_module.html


hfag.phys.ntu.edu.tw as reverse proxy to dayabay.phys.ntu.edu.tw
-------------------------------------------------------------------

#. Configured `--without-http_rewrite_module` as no PCRE on H.

::

	[root@hfag e]# date
	Thu Apr 25 18:57:58 GMT+8 2013

	[root@hfag e]# which nginx
	/data/usr/local/nginx/sbin/nginx

	[root@hfag e]# nginx -h
	nginx version: nginx/1.3.2
	Usage: nginx [-?hvVtq] [-s signal] [-c filename] [-p prefix] [-g directives]

	Options:
	  -?,-h         : this help
	  -v            : show version and exit
	  -V            : show version and configure options then exit
	  -t            : test configuration and exit
	  -q            : suppress non-error messages during configuration testing
	  -s signal     : send signal to a master process: stop, quit, reopen, reload
	  -p prefix     : set prefix path (default: /data/usr/local/nginx/)
	  -c filename   : set configuration file (default: conf/nginx.conf)
	  -g directives : set global directives out of configuration file


Setup reverse proxy with `nginx-edit` from H to C2 to subvert the `N -> C2` blockade that is again applied::

       location / {
              proxy_pass     http://dayabay.phys.ntu.edu.tw ;
              proxy_redirect http://dayabay.phys.ntu.edu.tw/ /;
              proxy_set_header  X-Real-IP  \$remote_addr;
        }

Switch the SVN source on N, spitting in the face of the network bstards that blocked me yet again::

	[blyth@belle7 e]$ svn switch --relocate http://dayabay.phys.ntu.edu.tw/repos/env http://hfag.phys.ntu.edu.tw:90/repos/env


SCGI Mounting
----------------

* http://wiki.nginx.org/HttpScgiModule

  * module first appeared in nginx-0.8.42


attempt to upgrade nginx on N
------------------------------

::

    [blyth@belle7 daeserver]$ sudo yum --enablerepo=epel info nginx
    Loaded plugins: kernel-module
    epel                      100% |=========================| 3.6 kB    00:00     
    epel/primary_db           100% |=========================| 3.1 MB    00:40     
    Installed Packages
    Name       : nginx
    Arch       : i386
    Version    : 0.6.39
    Release    : 4.el5
    Size       : 710 k
    Repo       : installed
    Summary    : Robust, small and high performance http and reverse proxy server
    URL        : http://nginx.net/
    License    : BSD
    Description: Nginx [engine x] is an HTTP(S) server, HTTP(S) reverse proxy and IMAP/POP3
               : proxy server written by Igor Sysoev.
               : 
               : One third party module, nginx-upstream-fair, has been added.

    Available Packages
    Name       : nginx
    Arch       : i386
    Version    : 0.8.55
    Release    : 3.el5
    Size       : 391 k
    Repo       : epel
    Summary    : Robust, small and high performance HTTP and reverse proxy server
    URL        : http://nginx.net/
    License    : BSD
    Description: Nginx [engine x] is an HTTP(S) server, HTTP(S) reverse proxy and IMAP/POP3
               : proxy server written by Igor Sysoev.



::

    [blyth@belle7 ~]$ sudo tail -18  /var/log/nginx/error.log
    2013/11/03 14:35:11 [error] 4410#0: *126288 open() "/usr/share/nginx/html/rootdoc/src/THistPainter.h.html" failed (2: No such file or directory), client: 5.10.83.73, server: _, request: "GET /rootdoc/src/THistPainter.h.html HTTP/1.1", host: "belle7.nuu.edu.tw"
    panic: MUTEX_LOCK (22) [op.c:352].
    2013/11/03 15:21:09 [emerg] 31049#0: eventfd() failed (38: Function not implemented)
    2013/11/03 15:21:09 [alert] 31011#0: worker process 31049 exited with fatal code 2 and can not be respawn
    2013/11/03 15:23:37 [notice] 19916#0: signal process started
    2013/11/03 15:23:37 [error] 19916#0: invalid PID number "" in "/var/run/nginx.pid"
    2013/11/03 15:24:03 [notice] 23282#0: signal process started
    2013/11/03 15:24:03 [error] 23282#0: invalid PID number "" in "/var/run/nginx.pid"
    2013/11/03 15:24:18 [emerg] 25263#0: eventfd() failed (38: Function not implemented)
    2013/11/03 15:24:18 [alert] 25255#0: worker process 25263 exited with fatal code 2 and can not be respawn
    2013/11/03 15:24:30 [notice] 26924#0: signal process started
    2013/11/03 15:24:30 [error] 26924#0: open() "/var/run/nginx.pid" failed (2: No such file or directory)
    2013/11/03 15:24:37 [notice] 27757#0: signal process started
    2013/11/03 15:24:37 [error] 27757#0: open() "/var/run/nginx.pid" failed (2: No such file or directory)
    2013/11/03 15:26:20 [emerg] 9847#0: eventfd() failed (38: Function not implemented)
    2013/11/03 15:26:20 [alert] 9843#0: worker process 9847 exited with fatal code 2 and can not be respawn
    2013/11/03 15:31:21 [emerg] 21271#0: eventfd() failed (38: Function not implemented)
    2013/11/03 15:31:21 [alert] 20776#0: worker process 21271 exited with fatal code 2 and can not be respawn
    [blyth@belle7 ~]$ 


* http://forum.nginx.org/read.php?2,150853,150853

nginx binary was compiled with file aio support (on another
host), but your system doesn't have proper interfaces (lacks
eventfd() syscall).

Solution is to recompile nginx without file aio support.

It's not compiled in by default, so just don't specify
--with-file-aio flag to configure.



Erase the broken yum/epel nginx
--------------------------------

First grab customized things::

    [blyth@belle7 ~]$ nginx-conf
    /etc/nginx/nginx.conf
    [blyth@belle7 ~]$ cp /etc/nginx/nginx.conf .
    [blyth@belle7 ~]$ cp /etc/nginx/users.txt .

Check what about to erase::

    [blyth@belle7 ~]$ rpm -ql nginx
    ...
    /etc/nginx
    /etc/nginx/fastcgi.conf
    /etc/nginx/fastcgi.conf.default
    /etc/nginx/fastcgi_params
    /etc/nginx/nginx.conf
    /etc/nginx/nginx.conf.default
    /etc/nginx/scgi_params
    ...

    [blyth@belle7 ~]$ sudo yum erase nginx
    ---> Package nginx.i386 0:0.8.55-3.el5 set to be erased
    warning: /etc/nginx/nginx.conf saved as /etc/nginx/nginx.conf.rpmsave
    Complete!

After::

    [blyth@belle7 ~]$ l /etc/nginx/
    total 16
    -rw-r--r-- 1 root root 5304 Nov  3 15:21 nginx.conf.rpmsave
    -rw-r--r-- 1 root root 3234 Feb 22  2013 nginx.conf.rpmnew
    -rw-r--r-- 1 root root   36 Jul  5  2010 users.txt

Build from source
------------------

::
  
    nginx-get
    nginx-build


Fail to follow symlinks
--------------------------

Following upgrade and a change of root the old one, nginx would not
follow symlinks.  Resolved by using the default htdocs and copying the 
links in there.



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

nginx-mode(){
   case $NODE_TAG in 
     WW|H|N) echo src ;;
          *) echo $(pkgr-cmd) ;;
   esac
}

#nginx-name(){ echo nginx-0.7.61 ; }
#nginx-url(){  echo http://sysoev.ru/nginx/$(nginx-name).tar.gz ; }
nginx-name(){ echo nginx-1.3.2 ; }
#nginx-name(){ echo nginx-0.8.55 ; }
nginx-url(){  echo http://nginx.org/download/$(nginx-name).tar.gz ; }

nginx-dir(){  echo $(local-base)/env/nginx/$(nginx-name) ; }
nginx-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(nginx-dir)) && mkdir -p $dir
   cd $dir
   local ans
   read -p "$msg consider using your systems packager for simplicitym or enter YES to proceed ... " ans
   [ "$ans" != "YES" ] && echo $msg OK skipping && return 
   [ ! -d "$(nginx-name)" ] && curl -L -O $(nginx-url) && tar zxvf $(nginx-name).tar.gz  
}
nginx-build(){
   cd $(nginx-dir)
   case $(uname -n) in 
     hfag.phys.ntu.edu.tw) ./configure --prefix=$(local-base)/nginx --without-http_rewrite_module  ;;
                        *) ./configure --prefix=$(local-base)/nginx ;;
   esac
   make
}

nginx-install(){
   cd $(nginx-dir)
   $SUDO make install
}



nginx-prefix(){
  case $(nginx-mode) in 
    port) echo /opt/local ;; 
    ipkg) echo /opt ;;
     src) echo $(local-base)/nginx ;;  
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
nginx-conf(){
   case $NODE_TAG in 
     WW|H|N) echo $(nginx-prefix)/conf/nginx.conf ;;
          *) echo $(nginx-confd $*)/nginx.conf ;;
   esac
}    


 ## Since version 0.6.7 the filename path is relative to directory of nginx configuration file nginx.conf, but not to nginx prefix 
nginx-users(){   echo $(nginx-prefix)/etc/nginx/users.txt ; }
nginx-pidpath(){ 
  case $(pkgr-cmd) in 
    port) echo $(nginx-prefix)/var/run/nginx/nginx.pid ;;
    ipkg) echo $(nginx-prefix)/var/nginx/run/nginx.pid ;;
     yum) echo $(nginx-prefix)/var/run/nginx.pid ;;
  esac
}    


# OK without sudo at IHEP    
nginx--(){ $SUDO $(nginx-sbin)/nginx $* ; }
nginx-sstop(){   nginx-- -s stop ; }
nginx-sstart(){  nginx--  ; }
nginx-srestart(){  
   nginx-sstop
   nginx-sstart
 }


 
nginx-pid(){     cat $(nginx-pidpath) 2>/dev/null ; }

nginx-stop(){    sudo kill -QUIT $(nginx-pid) ; }
nginx-start(){   
   local msg="=== $FUNCNAME :"
   local pid=$(nginx-pid)
   [ -n "$pid" ] && echo $msg looks like nginx is running already with pid $pid from pidfile $(nginx-pidpath) ... stop it first && return 0
   sudo /usr/sbin/nginx ; 
}

nginx-strings(){
   strings $(which nginx) | grep nginx.conf
}
nginx-info(){  cat << EOI
    nginx-pid : $(nginx-pid) 
EOI
}


nginx-syshtdocs(){
  echo /usr/share/nginx/html
}


nginx-htdocs(){ 
  case ${1:-$NODE_TAG} in 
     N|WW) echo $(nginx-prefix)/html ;; 
     *) echo $(nginx-eprefix)/share/nginx/html ;;
  esac
}

nginx-logd(){   
   case $(nginx-mode) in 
      src)  echo $(nginx-prefix)/logs ;;
     port)  echo $(pkgr-logd)/nginx  ;;
     ipkg)  echo $(nginx-prefix)/var/nginx/log ;;
      yum)  echo $(nginx-prefix)/var/log/nginx ;;
   esac  
}

nginx-cd(){     cd $(nginx-logd) ; }
nginx-log(){    echo $(nginx-logd)/sv.log ; }
nginx-elog(){   echo $(nginx-logd)/error.log ; }
nginx-alog(){   echo $(nginx-logd)/access.log ; }
nginx-tail(){   $SUDO tail -f $(nginx-log) ; }
nginx-etail(){  $SUDO tail -f $(nginx-elog) ; }
nginx-atail(){  $SUDO tail -f $(nginx-alog) ; }



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





nginx-edit(){  $SUDO vim $(nginx-conf) ; }
nginx-ps(){ ps aux | grep nginx ; }

# this is only for newer nginx
#nginx-stop(){
#    sudo nginx -s stop
#}

nginx-check(){
   $SUDO $(nginx-sbin)/nginx -c $(nginx-conf) -t 
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


nginx-links(){
  cd $(nginx-htdocs)
  $SUDO ln -s /home/blyth/env/muon_simulation muon_simulation
  $SUDO ln -s /usr/local/share/doc/gperftools-2.1 gperftools
  $SUDO ln -s /home/blyth/fast fast
  $SUDO ln -s /home/blyth/env/_build/dirhtml e
  $SUDO ln -s /data1/env/local/dyb/external/ROOT/5.26.00e_python2.7/i686-slc5-gcc41-dbg/root/htmldoc rootdoc
  $SUDO ln -s /data1/env/local/dyb/users/blyth/dbiscan/sphinx/_build/dirhtml dbiscan
  $SUDO ln -s /data1/env/local/dyb/NuWa-trunk/dybgaudi/Documentation/OfflineUserManual/tex/_build/dirhtml oum
  $SUDO ln -s /data1/env/local/dyb/NuWa-trunk/dybgaudi/Documentation/DoxyManual/dox doc
  $SUDO ln -s /data1/env/local/dyb/NuWa-trunk/dybgaudi/Documentation/OfflineUserManual/tex manual
}




