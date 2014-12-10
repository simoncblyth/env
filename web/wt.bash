# === func-gen- : web/wt fgp web/wt.bash fgn wt fgh web
wt-src(){      echo web/wt.bash ; }
wt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(wt-src)} ; }
wt-vi(){       vi $(wt-source) ; }
wt-env(){      elocal- ; }
wt-usage(){ cat << EOU

C++ Web Toolkit
=================

A recent addition to Geant4 is based in this.

http://www.webtoolkit.eu/wt



EOU
}
wt-dir(){ echo $(local-base)/env/web/web-wt ; }
wt-cd(){  cd $(wt-dir); }
wt-mate(){ mate $(wt-dir) ; }
wt-get(){
   local dir=$(dirname $(wt-dir)) &&  mkdir -p $dir && cd $dir

}
