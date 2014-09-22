# === func-gen- : numpy/npyreader fgp numpy/npyreader.bash fgn npyreader fgh numpy
npyreader-src(){      echo numpy/npyreader.bash ; }
npyreader-source(){   echo ${BASH_SOURCE:-$(env-home)/$(npyreader-src)} ; }
npyreader-vi(){       vi $(npyreader-source) ; }
npyreader-env(){      elocal- ; }
npyreader-usage(){ cat << EOU


* http://jcastellssala.wordpress.com/2014/02/01/npy-in-c/
* http://sourceforge.net/projects/kxtells.u/files/



EOU
}
npyreader-name(){ echo npyreader-0.01 ; }
npyreader-dir(){ echo $(local-base)/env/numpy/$(npyreader-name) ; }
npyreader-cd(){  cd $(npyreader-dir); }
npyreader-mate(){ mate $(npyreader-dir) ; }
npyreader-get(){
   local dir=$(dirname $(npyreader-dir)) &&  mkdir -p $dir && cd $dir

   local nam=$(npyreader-name)
   local tgz=$nam.tar.gz

   [ ! -f "$tgz" ] && curl -L -O http://downloads.sourceforge.net/project/kxtells.u/$tgz
   [ ! -d "$nam" ] && tar zxvf $tgz

}
