# === func-gen- : apache/apachebuild/modscgi fgp apache/apachebuild/modscgi.bash fgn modscgi fgh apache/apachebuild
modscgi-src(){      echo apache/apachebuild/modscgi.bash ; }
modscgi-source(){   echo ${BASH_SOURCE:-$(env-home)/$(modscgi-src)} ; }
modscgi-vi(){       vi $(modscgi-source) ; }
modscgi-env(){      elocal- ; }
modscgi-usage(){
  cat << EOU

MOD_SCGI
==========

* http://python.ca/scgi/
* http://python.ca/scgi/releases/
* http://docs.djangoproject.com/en/dev/howto/deployment/fastcgi/

Source::

   git clone http://quixote.ca/src/scgi.git


Installs
---------

D : system apache
~~~~~~~~~~~~~~~~~~~

::

   modscgi-
   modscgi-get

::

    (daeserver_env)delta:env blyth$ modscgi-install
    modscgi-install is a function
    modscgi-install () 
    { 
        local msg="=== $FUNCNAME :";
        [ -f "$(modscgi-so)" ] && echo $msg module is already installed at $(modscgi-so) && return 1;
        [ "$(which apxs)" == "" ] && echo $msg error no apxs : you may need to : sudo yum install httpd-devel && return 1;
        modscgi-cd apache2;
        type $FUNCNAME;
        apxs -c mod_scgi.c;
        sudo apxs -i -c mod_scgi.c
    }
    /usr/share/apr-1/build-1/libtool --silent --mode=compile /Applications/Xcode.app/Contents/Developer/Toolchains/OSX10.9.xctoolchain/usr/bin/cc    -DDARWIN -DSIGPROCMASK_SETS_THREAD_MASK -I/usr/local/include -I/usr/include/apache2  -I/usr/include/apr-1   -I/usr/include/apr-1   -c -o mod_scgi.lo mod_scgi.c && touch mod_scgi.slo
    env: /Applications/Xcode.app/Contents/Developer/Toolchains/OSX10.9.xctoolchain/usr/bin/cc: No such file or directory
    apxs:Error: Command failed with rc=65536




FUNCTIONS
----------

modscgi-src : $(modscgi-src)
modscgi-dir : $(modscgi-dir)

modscgi-ip  : $(modscgi-ip)
        normally SCGI server runs on same machine as the apache/lighttpd/nginx 
        that passes requests to it 

modscgi-port <name>
        port assigned to the named app 
        configure ports in one place for simplicity 

        For example :
           modscgi-port hg  : $(modscgi-port hg)
           modscgi-port dbi : $(modscgi-port dbi)



EOU
}
#modscgi-nam(){ echo scgi-1.13 ; }
modscgi-nam(){ echo scgi-1.14 ; }
modscgi-dir(){ echo $(local-base)/env/modscgi/$(modscgi-nam) ; }
modscgi-cd(){  cd $(modscgi-dir)/$1 ; }
modscgi-mate(){ mate $(modscgi-dir) ; }
modscgi-get(){
   local dir=$(dirname $(modscgi-dir)) &&  mkdir -p $dir && cd $dir
   local tgz=$(modscgi-nam).tar.gz
   [ ! -f "$tgz" ] && curl -O http://python.ca/scgi/releases/$tgz
   [ ! -d "$(modscgi-nam)" ] && tar zxvf $tgz
}
modscgi-so(){  echo $(apache-modulesdir)/mod_scgi.so ; }


modscgi-build(){
  modscgi-get
  modscgi-install
  modscgi-conf
}

modscgi-mavericks-apxs-kludge(){

   cat << EON

APXS config has a non-existant path, workaround  
is the below symlink 

* http://apple.stackexchange.com/questions/58186/how-to-compile-mod-wsgi-mod-fastcgi-etc-on-mountain-lion-mavericks-by-fixing

Suggests::

   sudo ln -s /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/ /Applications/Xcode.app/Contents/Developer/Toolchains/OSX10.9.xctoolchain

Before kludge::

    delta:~ blyth$ ll /Applications/Xcode.app/Contents/Developer/Toolchains/
    total 0
    drwxr-xr-x  3 root  wheel  102 Jan 16  2014 .
    drwxr-xr-x  4 root  wheel  136 Apr 15 16:05 XcodeDefault.xctoolchain
    drwxr-xr-x  9 root  wheel  306 Apr 15 16:05 ..

After::

    delta:~ blyth$ ll /Applications/Xcode.app/Contents/Developer/Toolchains/
    total 8
    drwxr-xr-x  4 root  wheel  136 Apr 15 16:05 XcodeDefault.xctoolchain
    drwxr-xr-x  9 root  wheel  306 Apr 15 16:05 ..
    lrwxr-xr-x  1 root  wheel   24 Sep 22 20:28 OSX10.9.xctoolchain -> XcodeDefault.xctoolchain
    drwxr-xr-x  4 root  wheel  136 Sep 22 20:28 .


EON

   sudo bash -c "cd /Applications/Xcode.app/Contents/Developer/Toolchains/ ; ln -s XcodeDefault.xctoolchain OSX10.9.xctoolchain "

}


modscgi-install(){
   local msg="=== $FUNCNAME :"
   [ -f "$(modscgi-so)" ] && echo $msg module is already installed at $(modscgi-so) && return 1 
   [ "$(which apxs)" == "" ]   && echo $msg error no apxs : you may need to : sudo yum install httpd-devel  && return 1

   modscgi-cd apache2
   type $FUNCNAME
   apxs -c mod_scgi.c
   sudo apxs -i -c mod_scgi.c
}




modscgi-apps(){ echo hg dbi runinfo ; }
modscgi-port(){  local-port $* ; }
modscgi-ip(){   echo 127.0.0.1 ; } 


modscgi-head-(){ cat << EOH
## $FUNCNAME 
##  http://quixote.python.ca/scgi.dev/doc/guide.html ... seems to work both with and without the trailing slash 
LoadModule scgi_module $(basename $(apache-modulesdir))/mod_scgi.so
EOH
}

modscgi-conf-(){
  local app=${1:-appname}  
  apache-
  cat << EOC
SCGIMount /$app $(modscgi-ip $app):$(local-port $app)
EOC
}

modscgi-conf(){
  local msg="=== $FUNCNAME :"
  modscgi-head-
  local app ; for app in $(modscgi-apps) ; do
     $FUNCNAME- $app
  done 
  echo $msg incorporate config similar to the above with : \"apache-edit\" 
}



