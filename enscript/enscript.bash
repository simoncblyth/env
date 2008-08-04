

enscript-usage(){
 cat << EOU

   Installed to cut down on noise in the Trac log and 
   associated performance hit 


EOU

}

enscript-env(){
 elocal-
}

enscript-name(){
  echo enscript-1.6.1
}

enscript-url(){
  echo http://ftp.gnu.org/pub/gnu/enscript/$(enscript-name).tar.gz
}

enscript-dir(){
  echo  $(local-system-base)/enscript
}

enscript-cd(){
  cd $(enscript-dir)
}

enscript-get(){
   local iwd=$PWD
   local dir=$(enscript-dir)
   $SUDO mkdir -p $dir && cd $dir

   local nam=$(enscript-name)
   local tgz=$nam.tar.gz

   [ ! -f $tgz ] && curl -O $(enscript-url)
   [ ! -d $nam ] && tar zxvf $tgz

   cd $iwd
}








