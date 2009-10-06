# === func-gen- : base/svman fgp base/svman.bash fgn svman fgh base
svman-src(){      echo base/svman.bash ; }
svman-source(){   echo ${BASH_SOURCE:-$(env-home)/$(svman-src)} ; }
svman-vi(){       vi $(svman-source) ; }
svman-env(){      elocal- ; }
svman-usage(){
  cat << EOU
     svman-src : $(svman-src)
     svman-dir : $(svman-dir)

     svman-get/update
         Getting and building the Supervisor manual

     svman-open

EOU
}
svman-mate(){ mate $(svman-dir) ; }
svman-url(){ echo http://svn.supervisord.org/supervisor_manual/trunk ; }
svman-dir(){ echo $(sv-;sv-dir)/manual ; }
svman-cd(){  cd $(svman-dir) ; }
svman-get(){
   local dir=$(svman-dir) && mkdir -p $(dirname $dir) && cd $(dirname $dir)
   svn co $(svman-url) manual
}
svman-update(){
   local msg="=== $FUNCNAME :"
   [ "$(which xsltproc)" == "" ] && echo $msg ABORT no xsltproc && return 1
   svman-cd
   svn up
   autoconf   
   ./configure
   make
}
svman-open(){ open $(svman-dir)/html/index.html ; }






