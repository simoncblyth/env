# === func-gen- : apache/apachebuild/modxsendfile fgp apache/apachebuild/modxsendfile.bash fgn modxsendfile fgh apache/apachebuild
modxsendfile-src(){      echo apache/apachebuild/modxsendfile.bash ; }
modxsendfile-source(){   echo ${BASH_SOURCE:-$(env-home)/$(modxsendfile-src)} ; }
modxsendfile-vi(){       vi $(modxsendfile-source) ; }
modxsendfile-env(){      elocal- ; }
modxsendfile-usage(){
  cat << EOU
     modxsendfile-src : $(modxsendfile-src)
     modxsendfile-dir : $(modxsendfile-dir)

     http://tn123.ath.cx/mod_xsendfile/

The installation added the line to httpd.conf :
     LoadModule xsendfile_module   libexec/apache2/mod_xsendfile.so

I added the directive :
     XSendFile on



EOU
}
modxsendfile-nam(){ echo mod_xsendfile-0.9 ; }
modxsendfile-dir(){ echo $(local-base)/env/modxsendfile/$(modxsendfile-nam) ; }
modxsendfile-url(){ echo http://tn123.ath.cx/mod_xsendfile/$(modxsendfile-nam).tar.gz ; }
modxsendfile-cd(){  cd $(modxsendfile-dir)/$1 ; }
modxsendfile-mate(){ mate $(modxsendfile-dir) ; }
modxsendfile-get(){
   local dir=$(dirname $(modxsendfile-dir)) &&  mkdir -p $dir && cd $dir
   local tgz=$(modxsendfile-nam).tar.gz
   [ ! -f "$tgz" ] && curl -O $(modxsendfile-url)
   [ ! -d "$(modxsendfile-nam)" ] && tar zxvf $tgz
}
modxsendfile-so(){  echo $(apache-modulesdir)/mod_scgi.so ; }


modxsendfile-build(){
  modxsendfile-get
  modxsendfile-install
  #modxsendfile-conf
}

modxsendfile-install(){
   local msg="=== $FUNCNAME :"
   [ -f "$(modxsendfile-so)" ] && echo $msg module is already installed at $(modxsendfile-so) && return 1 
   [ "$(which apxs)" == "" ]   && echo $msg error no apxs : you may need to : sudo yum install httpd-devel  && return 1

   modxsendfile-cd 
   sudo apxs -cia mod_xsendfile.c
}




