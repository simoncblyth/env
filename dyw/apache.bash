
##
##

apache-x(){ scp $HOME/$DYW_BASE/apache.bash ${1:-$TARGET_NODE}:$DYW_BASE ; }

# 
#   usage:
#
#   apache-get
#   apache-configure
#   apache-install
#   apache-start
#   apache-stop
#
#
#   note if installed as non-sudoer will live at 
#      http://localhost:8080
#   rather than
#      http://localhost
#
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

if [ "$LOCAL_NODE" == "g4pb" ]; then

  ## use the stock apache on g4pb 
  export APACHE_HTDOCS=/Library/WebServer/Documents
  
else

  APACHE_NAME=apache_1.3.37
  export APACHE_HOME=$LOCAL_BASE/apache/$APACHE_NAME
  export APACHE_HTDOCS=$APACHE_HOME/webserver/htdocs
  export PATH=$APACHE_HOME/sbin:$PATH
fi



apache-settings(){

  vi $APACHE_HOME/etc/httpd/{httpd,scb}.conf 
}


apache-get(){

  n=$APACHE_NAME
  tgz=$n.tar.gz
  url=http://apache.stu.edu.tw/httpd/$tgz

  cd $LOCAL_BASE

  test -d apache || ( $SUDO mkdir apache && $SUDO chown $USER apache )
  cd apache 
  
  test -f $tgz || curl -o $tgz $url

  test -f $n || mkdir $n
  cd $n

  test -d build || mkdir build
  tar -C build -zxvf ../$tgz 

}

apache-configure(){

   cd $APACHE_HOME/build/$APACHE_NAME
   ./configure --help
   layout="--prefix=$APACHE_HOME --with-layout=GNU --sysconfdir=$APACHE_HOME/etc/httpd --datadir=$APACHE_HOME/webserver --localstatedir=$APACHE_HOME/var "
   opts="--enable-module=most --enable-shared=max"
   $SUDO ./configure ${layout} ${opts}

}

apache-install(){
   cd $APACHE_HOME/build/$APACHE_NAME
   $SUDO make
   $SUDO make install
}

apache-start(){ 
	$SUDO apachectl start 
}

apache-stop(){  
	$SUDO apachectl stop 
}


