# === func-gen- : dj/djdep fgp dj/djdep.bash fgn djdep fgh dj
djdep-src(){      echo dj/djdep.bash ; }
djdep-source(){   echo ${BASH_SOURCE:-$(env-home)/$(djdep-src)} ; }
djdep-vi(){       vi $(djdep-source) ; }
djdep-env(){      elocal- ; }
djdep-usage(){
  cat << EOU


     runinfo + nginx + fastcgi shakedown ... 
          * find need to delete the socket prior to startup 




     djdep-src : $(djdep-src)
     djdep-dir : $(djdep-dir)

     http://docs.djangoproject.com/en/dev/howto/deployment/fastcgi/

     djdep-sv
         add the active dj-project:$(dj-project) to supervisor conf          

     djdep-media-ln
        plant link into apache-docroot enabling the serving
        of the django admin statics (stylesheets/images/javascript) with apache 

     djdep-media-cp
        if your apache config doesnt like the link use this instead 
        [Tue Oct 06 15:13:19 2009] [error] [client 140.112.102.77] Symbolic link not allowed: /var/www/html/media, referer: http://cms01.phys.ntu.edu.tw/runinfo/admin/

     djdep-media-test
         check media access ... if fails look at 
              /var/log/messages
              /var/log/httpd/error_log 


    djdep-run
          Error: No module named django_extensions

     dj-protocol : $(dj-protocol)
           are using scgi with apache and fcgi with lighttpd


EOU
}
djdep-dir(){ echo $(local-base)/env/dj/dj-djdep ; }
djdep-cd(){  cd $(djdep-dir); }
djdep-mate(){ mate $(djdep-dir) ; }
djdep-get(){
   #local dir=$(dirname $(djdep-dir)) &&  mkdir -p $dir && cd $dir
   echo -n
}

djdep-versions(){
   python -V
   echo ipython $(ipython -V)
   python -c "import mod_python as _ ; print 'mod_python:%s' % _.version "   
   python -c "import MySQLdb as _ ; print 'MySQLdb:%s' % _.__version__ "
   echo "select version() ; " | mysql-sh
   mysql_config --version 
   apachectl -v
   svn info $(dj-srcdir)
}

djdep-manage(){
   local iwd=$PWD
   cd $(dj-projdir)   
   case $1 in 
       shell)  sudo -u $(apache-user) $(dj-env-inline) ipython manage.py $* ;;
           *)  sudo -u $(apache-user) $(dj-env-inline)  python manage.py $* ;;
   esac
   cd $iwd
}

djdep-runserver(){
  cd $(dj-projdir)
  ENV_PRIVATE_PATH=$HOME/.bash_private python manage.py runserver 
}
djdep-env-inline(){  echo DJANGO_SETTINGS_MODULE=$(dj-settings-module) PYTHON_EGG_CACHE=$(djdep-eggcache-dir) ; }


djdep-notes(){
  cat << EON

   Proxying was used in order to simply keep generated 
   model files separate from the tweaked other files 

   NB 
     Override the locations with envvars 
          DJANGO_DIR     : $DJANGO_DIR
          DJANGO_APP     : $DJANGO_APP
          DJANGO_PROJECT : $DJANGO_PROJECT

   Deployments 

           http://belle7.nuu.edu.tw/dybsite/admin/
        N   : system python 2.4, mysql 5.0.24, MySQL_python-1.2.2, 
              system Mod Python , apache


           http://cms01.phys.ntu.edu.tw/dybsite/admin/
        C   : system python 2.3, mysql 4.1.22, MySQL_python  
           
               ===> admin pw needs resetting ...


        C2  :
           EXCLUDE FOR NOW AS PRIME REPO SERVER
                 which still uses source python 2.5
                 and source apache 2.0.63


        H :
            ancient machine ... not worth bothering with 


        G   :  
           
             port installed mysql 5.0.67

             Darwin difficulties ... need to be careful with system python  
             port installed python 2.6 and created virtualenv ~/v/djimg into
             which installed most of the preqs

           



EON
}




## admin site grabs 

djdep-admin-cp(){
  local msg="=== $FUNCNAME :"
  local rel=${1:-templates/admin/base_site.html}
  local srcd=$(dj-srcdir)/django/contrib/admin
  local dstd=$(dj-projdir)  

  echo $msg rel $rel srcd $srcd dstd $dstd 
  local path=$srcd/$rel
  local targ=$dstd/$rel
  [ ! -f "$path" ] && echo $msg ABORT no path $path && return 1
  local cmd="mkdir -p $(dirname $targ) &&  cp $path $targ "
    
  echo $msg $(dirname $path)
  ls -l $(dirname $path)
  echo $msg $(dirname $targ)
  ls -l $(dirname $targ)

  local ans
  read -p "$msg $cmd ... enter YES to proceed " ans
  [ "$ans" != "YES" ] && echo $msg skipping && return 0
  eval $cmd 
}




djdep-eggcache(){
   local cache=$(dj-eggcache-dir)
   [ "$cache" == "$HOME" ] && echo $msg cache is HOME:$HOME skipping && return 0

   echo $msg createing egg cache dir $cache
   sudo mkdir -p $cache
   apache- 
   apache-chown $cache
   sudo chcon -R -t httpd_sys_script_rw_t $cache
   ls -alZ $cache
}

djdep-selinux(){
local msg="=== $FUNCNAME :"

sudo chcon -R -t httpd_sys_content_t $(dj-srcdir)
sudo chcon -R -t httpd_sys_content_t $(dj-projdir) 
sudo chcon -R -t httpd_sys_content_t $(env-home)
}



#djdep-confname(){ echo 50-django.conf ; }
djdep-confname(){ echo $(dj-project).conf ; }

djdep-eggcache-prep(){
    local tmp=/tmp/env/$FUNCNAME/$USER && mkdir -p $tmp && echo $tmp
}

djdep-eggcache-dir(){ 
    case ${USER:-nobody} in 
      nobody|apache|www) echo /var/cache/dj ;; 
                      *) djdep-eggcache-prep  ;;  
    esac
}
djdep-deploy(){

   local msg="=== $FUNCNAME :"
   djdep-conf
   [ "$?" != "0" ] && echo $msg && return 1

   djdep-eggcache
   djdep-selinux

   private- 
   private-sync

   djdep-media-ln

   dj-syncdb
   #dj-test
}

djdep-server(){ 
   case ${1:-$NODE_TAG} in
      U|G) echo lighttpd  ;;
        C) echo nginx ;;
        *) echo apache ;;
   esac
}

djdep-conf(){
  local msg="=== $FUNCNAME :" 
  local tmp=/tmp/env/dj && mkdir -p $tmp 
  local conf=$tmp/$(djdep-confname)
  
  local server=$(djdep-server)
  djdep-location-$server- > $conf
  $server-
  cat $conf

  local confd=$($server-confd)
  [ ! -d "$confd" ] && echo $msg ABORT there is no confd : $confd  && return 1

  local cmd="sudo cp $conf $confd/$(basename $conf)"
  local ans
  read -p "$msg Proceed with : $cmd : enter YES to continue  " ans
  [ "$ans" != "YES" ] && echo $msg skipping && return 1
  eval $cmd

  echo $msg CHECK ACCESS LOGS ... IF GETTING PERMISSION DENIED YOU MAY NEED TO djdep-chownsocket
}


djdep-chownsocket(){
    local msg="=== $FUNCNAME :"
    local server=$(djdep-server)
    $server-
    local cmd="$server-chown $(djdep-socket)"
    echo $msg $cmd
    eval $cmd
}


djdep-location-apache-(){
  apache-
  private-
  cat << EOL


LoadModule python_module modules/mod_python.so

## each process only servers one request  ... huge performance hit 
## but good for development as means that code changes are immediately reflected 
MaxRequestsPerChild 1

<Location "$(dj-urlroot)/">
    SetHandler python-program
    PythonHandler django.core.handlers.modpython
    SetEnv ENV_PRIVATE_PATH $(USER=$(apache-user) private-path)    
    SetEnv DJANGO_SETTINGS_MODULE $(dj-settings-module)    
    SetEnv PYTHON_EGG_CACHE $(USER=$(apache-user) dj-eggcache-dir)
    PythonOption django.root $(dj-urlroot)
    PythonDebug On
</Location>

<Location "/media">
    SetHandler None
</Location>

<LocationMatch "\.(jpg|gif|png)$">
    SetHandler None
</LocationMatch>



EOL
# PythonPath "['$(dirname $(dj-projdir))', '$(dj-srcdir)'] + sys.path"
}


djdep-fcgiroot(){ echo /django.fcgi ; }
djdep-location-lighttpd-(){  cat << EOC


auth.debug = 1
auth.backend = "htdigest"
auth.backend.htdigest.userfile = "$(lighttpd-users)"

auth.require = ( "/protected/" =>
  (
   "method" => "digest",
   "realm" => "protected",
   "require" => "user=jamesbond"
  )
)

fastcgi.server = (
    "$(djdep-fcgiroot)" => (
           "main" => (
               "socket" => "$(djdep-socket)",
               "check-local" => "disable",
               "allow-x-send-file" => "enable" , 
                      )
                 ),
)

# The alias module is used to specify a special document-root for a given url-subset. 
alias.url += (
           "/media" => "$(python-site)/django/contrib/admin/media",  
)

url.rewrite-once += (
      "^(/media.*)$" => "\$1",
      "^/favicon\.ico$" => "/media/favicon.ico",
      "^/robots\.txt$" => "/robots.txt",
      "^(/.*)$" => "$(djdep-fcgiroot)\$1",
)

EOC
}




djdep-location-nginx-(){ cat << EOC

location /media {
     alias $(python-site)/django/contrib/admin/media;
}

location / {
      #auth_basic            "Restricted";
      #auth_basic_user_file  users.txt;
      fastcgi_pass   unix:$(djdep-socket);

     # http://code.djangoproject.com/wiki/ServerArrangements
     #     django needs PATH_INFO 
     #            with  SCRIPT_NAME it fails to match any urls 
     #
     fastcgi_param  PATH_INFO          \$fastcgi_script_name;
     # fastcgi_param  SCRIPT_NAME       \$fastcgi_script_name;

     fastcgi_param  QUERY_STRING       \$query_string;
     fastcgi_param  REQUEST_METHOD     \$request_method;
     fastcgi_param  CONTENT_TYPE       \$content_type;
     fastcgi_param  CONTENT_LENGTH     \$content_length;

     fastcgi_param  REQUEST_URI        \$request_uri;
     fastcgi_param  DOCUMENT_URI       \$document_uri;
     fastcgi_param  DOCUMENT_ROOT      \$document_root;
     fastcgi_param  SERVER_PROTOCOL    \$server_protocol;

     fastcgi_param  GATEWAY_INTERFACE  CGI/1.1;
     fastcgi_param  SERVER_SOFTWARE    nginx/\$nginx_version;

     fastcgi_param  REMOTE_ADDR        \$remote_addr;
     fastcgi_param  REMOTE_PORT        \$remote_port;
     fastcgi_param  SERVER_ADDR        \$server_addr;
     fastcgi_param  SERVER_PORT        \$server_port;
     fastcgi_param  SERVER_NAME        \$server_name;

     # PHP only, required if PHP was built with --enable-force-cgi-redirect
     fastcgi_param  REDIRECT_STATUS    200;

}

EOC
}



## non-embedded deployment with apache mod_scgi or mod_fastcgi ?  or lighttpd/nginx

djdep-socket(){    echo /tmp/$(dj-project).sock ; }
djdep-port(){      echo $(local-port $(dj-project)) ; }
djdep-host(){      echo $(modscgi-;modscgi-ip $(dj-project)) ; }

djdep-approach(){ echo unixdomain ; }
djdep-connection(){
   case $(djdep-approach) in 
      unixdomain) echo socket=$(djdep-socket) ;;
               *) echo host=$(djdep-host) port=$(djdep-port) ;; 
   esac  
}

## defaults ...  maxspare=5 minspare=2 maxchildren=50 from : django/core/servers/fastcgi.py
djdep-opts-fcgi(){ echo runfcgi protocol=fcgi $(djdep-connection)  daemonize=false maxspare=3 minspare=1 ; }
djdep-opts-scgi(){ echo runfcgi protocol=scgi $(djdep-connection)  daemonize=false maxspare=3 minspare=1 ; }

## interactive config check 

djdep-run-(){ cat << EOR
./manage.py $(djdep-opts-$(dj-protocol)) --verbosity=2
EOR
}

djdep-run(){   
     local msg="=== $FUNCNAME :"
     cd $(dj-projdir) ;  

     if [ -S "$(djdep-socket)" ]; then
       echo $msg maybe you need to delete the socket  $(djdep-socket)
       ls -l $(djdep-socket)
     else
       echo $msg no socket $(djdep-socket)
     fi

     local pmd="umask u=rwx,g=rwx,o=rwx" 
     echo $msg remove the mask entirely for this process to allow nginx-nobody to access socket  $pmd
     eval $pmd

     local cmd="$(djdep-run-)" ;
     echo $msg $cmd  ... from $PWD 
     eval $cmd
 }  

djdep-sv-(){    
   dj-
   cat << EOC
[program:$(dj-project)]
command=$(which python) $(dj-projdir)/manage.py $(djdep-opts-$(dj-protocol))
redirect_stderr=true
redirect_stdout=true
autostart=true
priority=999
environment=ENV_PRIVATE_PATH=$ENV_PRIVATE_PATH,ENV_HOME=$ENV_HOME,PYTHON_EGG_CACHE=$(djdep-eggcache-dir $USER)
user=$USER
EOC
}

## socket=tcp://127.0.0.1:$(djdep-port)
djdep-sv-fcgi-(){ dj- ; private- ; cat << EOC
[fcgi-program:$(dj-project)]
socket=unix://$(djdep-socket)
command=$(which python) $(djdep-runfcgi-path)
redirect_stderr=true
redirect_stdout=true
priority=999
environment=ENV_PRIVATE_PATH='$(private-path)',ENV_HOME='$ENV_HOME',PYTHONPATH='$(dj-projdir)',DJANGO_SETTINGS_MODULE='settings'
EOC
}



djdep-runfcgi-(){ cat << EOS
#!/usr/bin/env python
if __name__ == '__main__':
    from flup.server.fcgi_fork import WSGIServer
    from django.core.handlers.wsgi import WSGIHandler
    WSGIServer(WSGIHandler()).run()
EOS
}


djdep-runfcgi-path(){ echo $(dj-projdir)/runfcgi.py ; }

## supervisor hookup with fcgi control by supervisor
djdep-sv-fcgi(){
    local msg="=== $FUNCNAME :"     
    local path=$(djdep-runfcgi-path)
    [ ! -f "$path" ] && echo $msg creating $path && djdep-runfcgi- > $path

    sv-
    $FUNCNAME- | sv-plus $(dj-project).ini ; 
}


## supervisor hookup with manage.py handled fcgi
djdep-sv(){  
   sv-
   $FUNCNAME- | sv-plus $(dj-project).ini ; 
}


djdep-media(){ 
   dj-
   echo $(dj-srcdir)/contrib/admin/media
}

djdep-media-cp(){
   local msg="=== $FUNCNAME :"
   apache-
   local cmd="sudo cp -r $(djdep-media) $(apache-docroot) "
   echo $msg \"$cmd\"
   eval $cmd    
   apache-own $(apache-docroot)/media
}

djdep-media-ln(){
   local msg="=== $FUNCNAME :"
   apache-
   local docroot=$(apache-docroot)
   dj-
   local media=$(djdep-media)
   [ ! -d "$media" ] && echo $msg ABORT no such dir $media && return 1
   local cmd="sudo ln -sf $media  $docroot/media"
   echo $msg $cmd
   eval $cmd
}

djdep-media-test(){
   curl http://localhost/media/css/base.css
}


