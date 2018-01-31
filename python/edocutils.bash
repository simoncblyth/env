# === func-gen- : python/edocutils fgp python/edocutils.bash fgn edocutils fgh python
edocutils-src(){      echo python/edocutils.bash ; }
edocutils-source(){   echo ${BASH_SOURCE:-$(env-home)/$(edocutils-src)} ; }
edocutils-vi(){       vi $(edocutils-source) ; }
edocutils-env(){      elocal- ; }
edocutils-usage(){ cat << EOU

   http://docutils.sourceforge.net/

   see also docutils-



EOU
}
edocutils-dir(){ echo $(local-base)/env/python/edocutils ; }
edocutils-cd(){  cd $(edocutils-dir); }
edocutils-mate(){ mate $(edocutils-dir) ; }
edocutils-get(){
   local dir=$(dirname $(edocutils-dir)) &&  mkdir -p $dir && cd $dir
   local url=http://docutils.svn.sourceforge.net/viewvc/docutils/trunk/docutils/?view=tar
   local tgz=docutils-snapshot.tgz
   [ ! -f "$tgz" ] && curl -o $tgz $url
   [ ! -d docutils ] && tar zxvf $tgz
}
edocutils-build(){
  edocutils-cd
  python setup.py build
}
edocutils-install(){
  edocutils-cd
  $SSUDO python setup.py install
}

