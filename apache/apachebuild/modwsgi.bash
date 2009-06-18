modwsgi-src(){    echo apache/apachebuild/modwsgi.bash ; }
modwsgi-source(){ echo ${BASH_SOURCE:-$(env-home)/$(modwsgi-src)} ; }
modwsgi-vi(){     vi $(modwsgi-source) ; }
modwsgi-usage(){  
   cat << EOU

    http://code.google.com/p/modwsgi/
    http://code.google.com/p/modwsgi/source

  modwsgi-get
  modwsgi-configure

      nothing goes to the PREFIX, just writes the mod_wsgi.so directly to apache

      from the README:
            Whatever version of Python is used, it must have been compiled with support
            for multithreading. To avoid a measure of memory bloat with your Apache
 	    processes, Python should also have been compiled with shared library
 	    support enabled. The majority of Python binary packages for Linux systems
 	    are not compiled with shared library support enabled. You should therefore
 	    consider recompiling Python from source code with shared library support
 	    enabled.:0

  modwsgi-install


  modwsgi-app-test
       Follow along http://code.google.com/p/modwsgi/wiki/QuickConfigurationGuide` 
        Works remotely too ..
            http://belle7.nuu.edu.tw/myapp/


  ANCIENT FUNCS FROM ABORTED USAGE ATTEMPT circa 2007

  modwsgi-tracs-conf
        BLOCKER credentials are not passed from modwsgi into Trac, so cannot
                login to trac : due to this issue, moved to using modpython
  modwsgi-apache2-conf      add the LoadModule line to httpd.conf


EOU
}



modwsgi-env(){
   elocal-
   apache-
   python-
}


modwsgi-build(){
   local msg="=== $FUNCNAME :"
   local so=$(modwsgi-so)
   if [ ! -f $so ];  then
      echo $msg no $so ... attempt to create it 
      modwsgi-get
      modwsgi-configure
      modwsgi-install
   fi
   modwsgi-conf

}

modwsgi-nam(){ echo mod_wsgi-2.5 ; }
modwsgi-url(){ echo http://modwsgi.googlecode.com/files/$(modwsgi-nam).tar.gz ; }
modwsgi-fold(){ echo $(local-base)/env/modwsgi ; }
modwsgi-dir(){  echo $(modwsgi-fold)/$(modwsgi-nam) ; }
modwsgi-so(){   echo $(apache-modulesdir)/mod_wsgi.so ; }
modwsgi-cd(){   cd $(modwsgi-dir) ; }

modwsgi-get(){
   local fold=$(modwsgi-fold)
   local url=$(modwsgi-url)
   local nam=$(basename $url)
   mkdir -p $fold && cd $fold
   [ ! -f "$nam" ] && curl -O $url && tar zxvf $nam
}

modwsgi-configure(){
   [ "$(which apxs)" == "" ]   && echo $msg error no apxs : you may need to : sudo yum install httpd-devel  && return 1
   [ "$(which python)" == "" ] && echo $msg error no python && return 1
   cd $(modwsgi-dir)
  ./configure 
}

modwsgi-install(){
  cd $(modwsgi-dir)
   make
   sudo make install
}

modwsgi-conf-(){ 
  local dir=$(apache-modulesdir)
  local rel=$(basename $dir)
  echo "LoadModule wsgi_module $rel/mod_wsgi.so"
}

modwsgi-conf(){
  local msg="=== $FUNCNAME :"
  grep "$(modwsgi-conf-)" $(apache-conf)  && echo $msg  already hooked up to apache || echo $msg needs to be added to apache-conf
}


modwsgi-app-name(){ echo myapp ; }
modwsgi-app-test(){
    local msg="=== $FUNCNAME :"
    modwsgi-app
    modwsgi-app-conf
    [ "$(curl http://localhost/$(modwsgi-app-name)/)" == "Hello World!" ] && echo $msg SUCCEEDED || echo $msg FAILED
}
modwsgi-app-(){ cat << EOA
def application(environ, start_response):
    status = '200 OK'
    output = 'Hello World!'
    response_headers = [('Content-type', 'text/plain'),
                        ('Content-Length', str(len(output)))]
    start_response(status, response_headers)
    return [output]
EOA
}
modwsgi-app-path(){ echo $(apache-cgidir)/$(modwsgi-app-name).wsgi ; }
modwsgi-app(){
    local msg="=== $FUNCNAME :"
    local tmpd=/tmp/env/$FUNCNAME && mkdir -p $tmpd
    local path=$(modwsgi-app-path)
    local tmp=$tmpd/$(basename $path)
    [ -f "$path" ] && echo $msg path $path already exists && return 0
    modwsgi-app- > $tmp
    local cmd="sudo cp $tmp $path "
    echo $cmd
    eval $cmd
}
modwsgi-app-conf-(){ echo "WSGIScriptAlias /$(modwsgi-app-name) $(modwsgi-app-path)"; }
modwsgi-app-conf(){
    local msg="=== $FUNCNAME :"
    grep "$(modwsgi-app-conf-)" $(apache-conf)  && echo $msg already hooked up to apache || echo $msg needs to be added to apache-conf
}








modwsgi-tracs-conf2(){

  local userfile=$1
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
  local userfile=$1
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
  apache2-add-module wsgi
  apache2-settings
  apachectl configtest
}



