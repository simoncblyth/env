# === func-gen- : python/docutils fgp python/docutils.bash fgn docutils fgh python
docutils-src(){      echo python/docutils.bash ; }
docutils-source(){   echo ${BASH_SOURCE:-$(env-home)/$(docutils-src)} ; }
docutils-vi(){       vi $(docutils-source) ; }
docutils-env(){      elocal- ; }
docutils-usage(){ cat << EOU

   http://docutils.sourceforge.net/



EOU
}
docutils-dir(){ echo $(local-base)/env/python/docutils ; }
docutils-cd(){  cd $(docutils-dir); }
docutils-mate(){ mate $(docutils-dir) ; }
docutils-get(){
   local dir=$(dirname $(docutils-dir)) &&  mkdir -p $dir && cd $dir
   local url=http://docutils.svn.sourceforge.net/viewvc/docutils/trunk/docutils/?view=tar
   local tgz=docutils-snapshot.tgz
   [ ! -f "$tgz" ] && curl -o $tgz $url
   [ ! -d docutils ] && tar zxvf $tgz
}
docutils-build(){
  docutils-cd
  python setup.py build
}
docutils-install(){
  docutils-cd
  $SSUDO python setup.py install
}

