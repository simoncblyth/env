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
modwsgi-app-path(){ echo $(apache-cgidir)/${1:-$(modwsgi-app-name)}.wsgi ; }
modwsgi-app-test(){
    local msg="=== $FUNCNAME :"
    local tmpd=/tmp/env/$FUNCNAME && mkdir -p $tmpd
    local path=$(modwsgi-app-path)
    local tmp=$tmpd/$(basename $path)
    [ -f "$path" ] && echo $msg path $path already exists && return 0
    echo $msg writing $tmp
    modwsgi-app- > $tmp
    modwsgi-deploy $tmp  
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

modwsgi-app-conf-(){ 
   local name=${1:-$(modwsgi-app-name)}
   echo "WSGIScriptAlias /$name $(modwsgi-app-path $name)"
}
modwsgi-app-conf(){
    local msg="=== $FUNCNAME :"
    local name=$1
    local conf=$(modwsgi-app-conf- $name)
    grep "$conf" $(apache-conf)  && echo $msg \"$conf\" already hooked up to apache || echo $msg \"$conf\" needs to be added to apache-conf
}



modwsgi-baseline-dir(){ echo $(local-base $*)/env/BASELINE ; } 
modwsgi-baseline(){
   local msg="=== $FUNCNAME :"
   [ "$(which virtualenv)" != "/usr/bin/virtualenv" ] && echo $msg unexpected virtualenv path && return 1
   echo $msg creating baseline virtualenv $dir as recommended : http://code.google.com/p/modwsgi/wiki/VirtualEnvironments
   local dir=$($FUNCNAME-dir)
   cd $(dirname $dir)
   virtualenv --no-site-packages $(basename $dir)

   $FUNCNAME-chcon $dir
}

modwsgi-baseline-chcon(){
   local dir=${1:-$(modwsgi-baseline-dir)}
   [ "$(which chcon)" == "" ] && echo $msg no chcon skip selinux labelling && return 0
   local cmd="sudo chcon -h -R -t httpd_sys_content_t $dir "
   echo $cmd
   eval $cmd
}

modwsgi-hmac-kludge(){
   local msg="=== $FUNCNAME :"
   local target=${1:-N}
   [ "$NODE_TAG" != "G" ] && echo $msg this must be done from G && return 1
   scp /opt/local/lib/python2.5/hmac.py $target:$(modwsgi-baseline-dir $target)/lib/python2.4/
}

modwsgi-hmac-test(){ $FUNCNAME- | python ; }
modwsgi-hmac-test-(){ cat << EOT
import pkg_resources
pkg_resources.get_distribution('Beaker').version
import beaker.session
print beaker.session.sha1 
import hmac
print hmac.new('test', 'test', beaker.session.sha1).hexdigest()
print hmac.__file__
EOT
}



modwsgi-virtualenv-(){
   echo "WSGIPythonHome $(modwsgi-baseline-dir)"
}

modwsgi-virtualenv(){
    local msg="=== $FUNCNAME :"

    local base=$(modwsgi-baseline-dir)
    [ ! -d "$base" ] && modwsgi-baseline || echo $msg baseline dir $base already exists 

    local conf=$(modwsgi-virtualenv- )
    grep "$conf" $(apache-conf)  
    local rc=$?
    case $rc in  
       0) echo $msg \"$conf\" already hooked up to apache ;;
       *) echo $msg \"$conf\" needs to be added to apache-conf && return $rc ;;
    esac
}

modwsgi-deploy(){
    local msg="=== $FUNCNAME :"
    local tmp=$1
    local base=$(basename $tmp)
    local name=${base/.*}  

    [ ! -f "$tmp" ] && echo $msg ABORT no such path $tmp && return 1
    [ "$name.wsgi" != "$base" ] && echo $msg ABORT path should end with .wsgi && return 1 

    local path=$(modwsgi-app-path $name)
  
    if [ -f "$path" ]; then
       echo $msg path $path is already present .. will overwrite it 
    fi 

    local cmd="sudo cp -f $tmp $path "
    echo $cmd
    eval $cmd
     

    modwsgi-app-conf $name  
    modwsgi-virtualenv 

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



