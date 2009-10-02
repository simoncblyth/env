# === func-gen- : apache/apachebuild/modscgi fgp apache/apachebuild/modscgi.bash fgn modscgi fgh apache/apachebuild
modscgi-src(){      echo apache/apachebuild/modscgi.bash ; }
modscgi-source(){   echo ${BASH_SOURCE:-$(env-home)/$(modscgi-src)} ; }
modscgi-vi(){       vi $(modscgi-source) ; }
modscgi-env(){      elocal- ; }
modscgi-usage(){
  cat << EOU
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

    http://python.ca/scgi/
    http://python.ca/scgi/releases/
    http://docs.djangoproject.com/en/dev/howto/deployment/fastcgi/

    git clone http://quixote.ca/src/scgi.git


EOU
}
modscgi-nam(){ echo scgi-1.13 ; }
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

modscgi-install(){
   local msg="=== $FUNCNAME :"
   [ -f "$(modscgi-so)" ] && echo $msg module is already installed at $(modscgi-so) && return 1 
   [ "$(which apxs)" == "" ]   && echo $msg error no apxs : you may need to : sudo yum install httpd-devel  && return 1

   modscgi-cd apache2
   type $FUNCNAME
   apxs -c mod_scgi.c
   sudo apxs -i -c mod_scgi.c
}




modscgi-apps(){ echo hg dbi ; }
modscgi-port(){ 
   case $1 in 
     hg) echo 5000 ;;
    dbi) echo 6000 ;;
      *) echo 7000 ;;
   esac  
}

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
SCGIMount /$app $(modscgi-ip $app):$(modscgi-port $app)
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



