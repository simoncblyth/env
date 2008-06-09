#
#   note if installed as non-sudoer will live at 
#      http://localhost:PORT
#
#   where PORT is the port specified by the "Listen" directive in the
#   httpd.conf file
#
#   the usual issue is :
#
#       http://grid1.phys.ntu.edu.tw:8080/autovalidation/
#   403 Forbidden : You don't have permission to access /autovalidation/ on this server.
#
#   fix this by setting permissions at both system and apache levels:
#
#  1) G1> cd $USER_BASE ; chmod -R go+rx jobs
#  2)  P>  putting a Directory entry into httpd.conf :   vi $APACHE_HOME/etc/httpd/httpd.conf
#
#
# 

apache2-usage(){

cat << EOU

  TODO...  look in the right place for logs on OSX
      /var/log/apache2


#   usage:
#
#   apache2-get
#   apache2-configure
#   apache2-install
#
#   apache2-setport
#   apache2-settings
#   apache2-settings-cmd
#   apache2-add-module
#
#   apache2-start
#   apache2-stop
#
#   apache2-ps
#   apache2-error-log
#   apache2-access-log
#
EOU

}



modpython-(){    . $ENV_HOME/apache/modpython.bash && modpython-env $* ; } 
modwsgi-use-(){  . $ENV_HOME/apache/modwsgi-use.bash && modwsgi-use-env $* ; } 
modwsgi-(){      . $ENV_HOME/apache/modwsgi.bash && modwsgi-env $* ; } 



apache2-env(){

   elocal-

   local APACHE2_NAME=httpd-2.0.59
   local APACHE2_ABBREV=apache2

   case $NODE_TAG in 
      G) APACHE2_USER=www ;;
	  H) APACHE2_USER=apache ;;
      *) APACHE2_USER=apache ;;
	esac

    export APACHE2_USER	  
    export APACHE2_HOME=$LOCAL_BASE/$APACHE2_ABBREV/$APACHE2_NAME
    export APACHE2_HOME_H=$LOCAL_BASE_H/$APACHE2_ABBREV/$APACHE2_NAME

   ##
   ## due to use of stock Leopard apache2 need to divide :
   ##     APACHE2_HOME : where apache was built
   ##     APACHE2_BASE : where apache is configured from 
   ##
   ##  these are the same in the self-made apache case on Tiger/Linux
   ##




   if [ "$NODE_APPROACH" == "stock" ]; then 
      export APACHE2_BASE=/private
      export APACHE2_LOCAL=etc/apache2/local
      export ASUDO=$SUDO
   else
      export APACHE2_BASE=$APACHE2_HOME
      export APACHE2_LOCAL=etc/apache2 
      export ASUDO=
      export PATH=$APACHE2_HOME/sbin:$PATH
      ##APACHE2_ENV=$APACHE2_HOME/sbin/envvars   not used ?
   fi

   export APACHE2_BUILD=$LOCAL_BASE/$APACHE2_ABBREV/build/$APACHE2_NAME

   if [ "$NODE_APPROACH" == "stock" ]; then
     APACHE2_CONF=/private/etc/apache2/httpd.conf
     APACHE2_HTDOCS=/Library/WebServer/Documents
     APACHE2_SO=libexec/apache2
   else
     APACHE2_CONF=$APACHE2_HOME/etc/apache2/httpd.conf
     APACHE2_HTDOCS=$APACHE2_HOME/share/apache2/htdocs
     APACHE2_SO=libexec 
   fi

   export APACHE2_CONF
   export APACHE2_PORT=6060

   export APACHE2_HTDOCS
   export APACHE2_HTDOCS_H=$APACHE2_HOME_H/share/apache2/htdocs
   export APACHE2_XSLT=$APACHE2_HTDOCS/resources/xslt

}



apache2-get(){

  n=$APACHE2_NAME
  a=$APACHE2_ABBREV

  tgz=$n.tar.gz
  url=http://ftp.mirror.tw/pub/apache/httpd/$tgz

  cd $LOCAL_BASE
  test -d $a || ( $SUDO mkdir $a && $SUDO chown $USER $a )
  cd $a 
  test -f $tgz || curl -o $tgz $url
  test -d build || mkdir build
  test -d build/$n || tar -C build -zxvf $tgz 
}


## invoked with sudo bash -lc apache2-configure

apache2-configure(){
   cd $APACHE2_BUILD

   ##
   ## http://www.devshed.com/c/a/Apache/Building-Apache-the-Way-You-Want-It/3/
   ##

   ## ooops this does not include mod_proxy
   ##opts="--enable-mods-shared=most "
   opts="--enable-mods-shared=all --enable-proxy=shared "


   $ASUDO ./configure --help
   layout="--prefix=$APACHE2_HOME --enable-layout=GNU "

   $ASUDO ./configure ${layout} ${opts}
}

apache2-install(){
   cd $APACHE2_BUILD
   $ASUDO make
   $ASUDO make install
}



apache2-setup(){
  apache2-set User www
  sudo chown -R www:www $APACHE2_HTDOCS
}




apache2-set(){
  
   #
   # Usage example:
   #      apache2-set User www
   #
   
   qwn=${1:-dummy}
   val=${2:-dummy}
   regx="s/^($qwn\s*)(\S*)\$/\${1}$val/g"
   echo set the quantity $qwn to value $val in $APACHE2_CONF ... $regx 
   echo $ASUDO perl -pi.orig -e $regx  $APACHE2_CONF
        $ASUDO perl -pi.orig -e $regx  $APACHE2_CONF
   diff  $APACHE2_CONF{.orig,} 
}

apache2-setport(){
   port=${1:-$APACHE2_PORT}
   regx="s/^(Listen\s*)(\S*)\$/\${1}$port/g"
   echo set the port to $port in $APACHE2_CONF ... $regx 
   echo $ASUDO perl -pi.orig -e $regx  $APACHE2_CONF
        $ASUDO perl -pi.orig -e $regx  $APACHE2_CONF
   diff  $APACHE2_CONF{.orig,} 
}
  
apache2-settings(){ 
  $ASUDO vi $APACHE2_HOME/etc/apache2/{httpd,trac,svn}.conf 
}
apache2-settings-cmd(){ 
  echo $ASUDO vi $APACHE2_HOME/etc/apache2/httpd.conf 
}


apache2-start(){ 
	apache2-setport $APACHE2_PORT 
	$ASUDO $APACHE2_HOME/sbin/apachectl start 
}

apache2-stop(){  
	$ASUDO $APACHE2_HOME/sbin/apachectl stop 
}

apache2-service-info(){
   cd /etc/init.d  && ls -alst ap*
}

apache2-service-setup(){
#
# considerations ...
#    everything is coming thru apache ... svn + trac 
#    sudo groups apache ... verified the "apache" group to exist 
#	
#	
#  1) stop apache2 (running as blyth)...   
#     	 apachectl stop
#  2) ammend the httpd.conf 
#        apache2-set User  apache
#        apache2-set Group apache
#  3) switch ownership:
#   	sudo chown -R apache $SCM_FOLD
#   	sudo chgrp -R apache $SCM_FOLD
#	
#  4) start apache2 ... NB must now use sudo as it needs to run as "apache"	
#        sudo apachectl start 
#
#    apache running OK and writing logs OK... but pysqlite problem
#    condistent with lack of LD_LIBRARY_PATH containing $SQLITE_HOME/lib	
#
#  5) allow the "apache" user to find sqlite libs without fooling with
#       LD_LIBRARY_PATH  of the "apache" user (is that possible ? they dony
#      have a login)
#	
#            sudo vi  /etc/ld.so.conf
#            add /data/usr/local/sqlite/sqlite-3.3.16/lib to /etc/ld.so.conf
#        sudo /sbin/ldconfig
#
#  6)  
#      	sudo apachectl stop
#      	sudo apachectl start
#
#      succeeds to run apache2 as the apache user ... and  trac works 
#	
#   7)
#      switch to apache2 as a service :
#
#          sudo apachectl stop
#          apache2-service start 
#
#
#

     [ -d /etc/init.d/ ] || ( echo "redhat specific ??  /sbin/service /etc/init.d wont work here " && return 1 )  
	cd /etc/init.d && sudo rm -f apache2 && sudo ln -s $APACHE2_HOME/sbin/apachectl apache2 && pwd && ls -l ap* 

}


apache2-service(){

   ##
   ##  normal "apachectl start" runs as "blyth" who owns /var/scm ... so no problem ..
   ##
   ##    
   ##
   ##
   ## current apache2 config runs as "nobody" 
   ##  giving permission problems to write to /var/scm/tracs/env/log/trac.log
   ##  config s "User" "Group" in httpd.conf
	echo "commands are passed thru to apachectl running as a service ... so can: start stop restart configtest ... " 
    sudo /sbin/service apache2 $* 	
}

apache-service(){
	echo "commands are passed thru to apachectl running as a service ... so can: start stop restart configtest ... " 
    sudo /sbin/service apache $* 	
}


apache2-error-log(){ tail -100 $APACHE2_HOME/var/apache2/log/error_log ; }
apache2-access-log(){ tail -100 $APACHE2_HOME/var/apache2/log/access_log ; }


apache2-favicon-get(){
  durl=http://grid1.phys.ntu.edu.tw:6060/tracs	
  uurl=${1:-$durl}
  file=favicon.ico
  cd $APACHE2_HTDOCS
  curl -o $file $uurl/$file
}

apache2-ps(){
  if [ "$LOCAL_ARCH" == "Darwin" ]; then
    psopt="-aux"  
  else
	psopt="-ef"
  fi
  ps $psopt | grep httpd
  
}


apache2-add-module(){
  name=${1:-dummy}
  [ "$name" == "dummy" ] && echo need parameter with the name of the module && return
  perl $SCM_HOME/apache-load-module.pl  $APACHE2_CONF $name add
}

apache2-conf-connect(){

  local pconf=${1:-dummy-connect} 

  ## stock Leopard apache2 using absolute paths /etc/apache2/blah/blah/
  ## but Tiger/Linux use relative starting with etc/apache2/blah..
  ## 
  if [ "$NODE_APPROACH" == "stock" ]; then
     local conf=$APACHE2_BASE/$pconf
  else
     local conf=$pconf
  fi 

  echo ============== add the Include of the $conf into $APACHE2_CONF if not there already
  $ASUDO bash -lc "grep $conf $APACHE2_CONF  || $ASUDO echo Include $conf  >> $APACHE2_CONF  "

  echo ============= tail -10 $APACHE2_CONF
  tail -10 $APACHE2_CONF 

}





apache2-open(){

  url=http://$TARGET_HOST:$TARGET_PORT
  echo try to open url:$url 

  if [ "$LOCAL_ARCH" == "Darwin" ]; then
	  open $url
  fi	  

}
