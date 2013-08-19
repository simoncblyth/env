# === func-gen- : tools/kcachegrind fgp tools/kcachegrind.bash fgn kcachegrind fgh tools
kcachegrind-src(){      echo tools/kcachegrind.bash ; }
kcachegrind-source(){   echo ${BASH_SOURCE:-$(env-home)/$(kcachegrind-src)} ; }
kcachegrind-vi(){       vi $(kcachegrind-source) ; }
kcachegrind-env(){      elocal- ; }
kcachegrind-usage(){ cat << EOU

http://kcachegrind.sourceforge.net/html/Home.html  

::

    [blyth@belle7 kcachegrind-0.7.4]$ vi README
    [blyth@belle7 kcachegrind-0.7.4]$ mkdir build
    [blyth@belle7 kcachegrind-0.7.4]$ cd build
    [blyth@belle7 build]$ /usr/lib/qt4/bin/qmake  ../qcg.pro
    Project ERROR: QCachegrind requires Qt 4.4 or greater


On belle7 qt4 is 4.2.1, 
skipping the version check fails in compilation


EOU
}
kcachegrind-dir(){ echo $(local-base)/env/tools/$(kcachegrind-name) ; }
kcachegrind-cd(){  cd $(kcachegrind-dir); }
kcachegrind-name(){ echo kcachegrind-0.7.4 ; }
kcachegrind-get(){
   local dir=$(dirname $(kcachegrind-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(kcachegrind-name)
   local tgz=$nam.tar.gz
   local url=http://kcachegrind.sourceforge.net/$tgz

   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz

}
