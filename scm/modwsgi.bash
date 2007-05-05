#
#
#  modwsgi-tracs-conf
#  modwsgi-app
#  modwsgi-apache2-conf      add the LoadModule line to httpd.conf
#
#  modwsgi-get
#  modwsgi-configure
#  modwsgi-install
#
#
#
# some modwsgi/trac issues :
#
#   1) SOLVED  http://grid1.phys.ntu.edu.tw:6060/trac
#        safari says "could not connect to server"
#      and in apache2-error-log :
#
# [Wed Apr 25 15:11:07 2007] [notice] mod_wsgi (pid=28604): Cleanup interpreter
# [Wed Apr 25 15:11:07 2007] [notice] mod_wsgi (pid=12999): Cleanup interpreter
# [Wed Apr 25 15:11:07 2007] [notice] mod_wsgi (pid=12997): Cleanup interpreter
# [Wed Apr 25 15:11:07 2007] [notice] mod_wsgi (pid=12995): Cleanup interpreter
# [Wed Apr 25 15:11:07 2007] [notice] mod_wsgi (pid=28605): Cleanup interpreter
# [Wed Apr 25 15:11:07 2007] [notice] SIGHUP received.  Attempting to restart
# [Wed Apr 25 15:11:07 2007] [notice] mod_wsgi (pid=28600): Terminating Python.
# [Wed Apr 25 15:11:07 2007] [notice] seg fault or similar nasty error detected in the parent process 
#
#      but  it works after a stop and start of apache2 rather an just a
#      restart ???
#
#   2) BLOCKER   credentials are not passed from modwsgi into Trac, so cannot
#       login to trac
#       due to this issue, moved to using modpython
#
#
#
#
modwsgi-x(){ scp $SCM_HOME/modwsgi.bash ${1:-$TARGET_TAG}:$SCM_BASE; }
modwsgi-i(){ . $SCM_HOME/modwsgi.bash ; }




modwsgi-tracs-conf2(){

## this is not working, trac is not getting the credentials
  userfile=$1


#
#http://www.modpython.org/pipermail/mod_python/2006-December/022841.html
## Nb changes here must be made in tandem with scm.bash::scm-create
cat << EOC

<LocationMatch ^/tracs/([^/]+)>
   SetHandler mod_wsgi
   #SetHandler wsgi-script
   SetEnv mod_wsgi.application application
   #SetEnv mod_wsgi.interpreter myapplication
   SetEnv mod_wsgi.directory   $SCM_FOLD/tracs/\$1/apache
   Options +ExecCGI
</LocationMatch>

# WSGIScriptAliasMatch ^/tracs/([^/]+) $SCM_FOLD/tracs/\$1/apache/\$1.wsgi
# WSGIPassAuthorization On

<LocationMatch ^/tracs/[^/]+/login>
   AuthType Basic
   AuthName "svn-tracs"
   AuthUserFile $userfile
   Require valid-user
</LocationMatch>

<DirectoryMatch ^$SCM_FOLD/tracs/([^/]+)/apache>
   Order deny,allow
   Allow from all
</DirectoryMatch>


EOC
} 




modwsgi-tracs-conf(){

## this is not working, trac is not getting the credentials
  userfile=$1

## Nb changes here must be made in tandem with scm.bash::scm-create
cat << EOC
<LocationMatch ^/tracs/[^/]+/login>
   AuthType Basic
   AuthName "svn-tracs"
   AuthUserFile $userfile
   Require valid-user
</LocationMatch>

WSGIScriptAliasMatch ^/tracs/([^/]+) $SCM_FOLD/tracs/\$1/apache/\$1.wsgi
WSGIPassAuthorization On

<DirectoryMatch ^$SCM_FOLD/tracs/([^/]+)/apache>
   Order deny,allow
   Allow from all
</DirectoryMatch>


EOC
} 




modwsgi-apache2-conf(){
  
  # add a line like this to httpd.conf
  #  "LoadModule wsgi_module        libexec/mod_wsgi.so"
  #

  apache2-add-module wsgi
  apache2-settings

  apachectl configtest


  # then  
  #   $APACHE2_HOME/sbin/apachectl restart 
  #      apache2-error-log 
  #  should  say:
  #
  # [Tue Apr 24 18:49:44 2007] [notice] mod_wsgi: Initializing Python.
  # [Tue Apr 24 18:49:44 2007] [notice] mod_wsgi (pid=28601): Attach interpreter ''.
  # [Tue Apr 24 18:49:44 2007] [notice] mod_wsgi (pid=28602): Attach interpreter ''.
  # [Tue Apr 24 18:49:44 2007] [notice] mod_wsgi (pid=28603): Attach interpreter ''.
  # [Tue Apr 24 18:49:44 2007] [notice] mod_wsgi (pid=28604): Attach interpreter ''.
  # [Tue Apr 24 18:49:44 2007] [notice] mod_wsgi (pid=28605): Attach interpreter ''.
  # [Tue Apr 24 18:49:44 2007] [notice] Apache/2.0.59 (Unix) DAV/2 SVN/1.4.0 mod_wsgi/1.0-TRUNK Python/2.5.1 configured -- resuming normal operations
  #
 
}




modwsgi-get(){

  ## http://code.google.com/p/modwsgi/
  ## http://code.google.com/p/modwsgi/source

  ##  repository not accessible so :
  ## [dayabaysoft@grid1 build]$ scp -r 20070424 H:$LOCAL_BASE_H/mod_wsgi/build/
  ##  and set the "last" link manually

   cd $LOCAL_BASE
   test -d mod_wsgi || ( $SUDO mkdir mod_wsgi  && $SUDO chown $USER mod_wsgi )
   cd mod_wsgi  
    
   test -d build || mkdir build
   refdef=$(date  +"%Y%m%d")

   svn checkout http://modwsgi.googlecode.com/svn/trunk/ build/$refdef 
   cd build && ln -s $refdef last 

}

modwsgi-configure(){
	    
   refdef=$(date  +"%Y%m%d")
   cd $LOCAL_BASE/mod_wsgi/build/last
  ./configure --prefix=$LOCAL_BASE/mod_wsgi/$refdef --with-apxs=$APACHE2_HOME/sbin/apxs  --with-python=$PYTHON_HOME/bin/python

# nothing goes to the PREFIX, just writes the mod_wsgi.so directly to apache
#
# from the README:
#   Whatever version of Python is used, it must have been compiled with support
#   for multithreading. To avoid a measure of memory bloat with your Apache
#	processes, Python should also have been compiled with shared library
#	support enabled. The majority of Python binary packages for Linux systems
#	are not compiled with shared library support enabled. You should therefore
#	consider recompiling Python from source code with shared library support
#	enabled.:0

#

}

modwsgi-install(){
	
   #refdef=$(date  +"%Y%m%d")

   cd $LOCAL_BASE/mod_wsgi/build/last
   make
   make install
   
   #cd $LOCAL_BASE/mod_wsgi/
   #ln -s $refdef last 
  
    echo ============= this creates $APACHE2_HOME/libexec/mod_wsgi.so
	ls -alst $APACHE2_HOME/libexec/mod_wsgi.so
}




