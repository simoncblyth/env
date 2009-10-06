# === func-gen- : base/svdev fgp base/svdev.bash fgn svdev fgh base
svdev-src(){      echo base/svdev.bash ; }
svdev-source(){   echo ${BASH_SOURCE:-$(env-home)/$(svdev-src)} ; }
svdev-vi(){       vi $(svdev-source) ; }
svdev-env(){      elocal- ; }
svdev-usage(){
  cat << EOU
     svdev-src : $(svdev-src)
     svdev-dir : $(svdev-dir)


EOU
}
svdev-mate(){ mate $(svdev-dir) ; }
svdev-url(){ echo http://svn.supervisord.org/supervisor/trunk ; }
svdev-dir(){ echo $(sv-;sv-dir)/dev ; }
svdev-mate(){ mate $(svdev-dir) ; }
svdev-cd(){  cd $(svdev-dir) ; }
svdev-get(){
   local dir=$(svdev-dir) && mkdir -p $(dirname $dir) && cd $(dirname $dir)
   svn co $(svdev-url) dev
}
svdev-log(){ svn log $(svdev-dir) ;  }



