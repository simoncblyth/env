# === func-gen- : python/distribute fgp python/distribute.bash fgn distribute fgh python
distribute-src(){      echo python/distribute.bash ; }
distribute-source(){   echo ${BASH_SOURCE:-$(env-home)/$(distribute-src)} ; }
distribute-vi(){       vi $(distribute-source) ; }
distribute-env(){      elocal- ; }
distribute-usage(){ cat << EOU

distribute forks+fixes setuptools
==================================

https://pypi.python.org/pypi/distribute



EOU
}
distribute-nam(){ echo distribute-0.6.36 ; }
distribute-dir(){ echo $(local-base)/env/python/$(distribute-nam) ; }
distribute-cd(){  cd $(distribute-dir); }
distribute-mate(){ mate $(distribute-dir) ; }
distribute-get(){
   local dir=$(dirname $(distribute-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(distribute-nam)
   local tgz=$nam.tar.gz
   local url=http://pypi.python.org/packages/source/d/distribute/$tgz
   [ ! -f "$tgz" ] && curl -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz
}
distribute-install(){
   distribute-cd
   echo consider eradicating the toxic setuptools from your system first 
   echo with your python of choice run : python setup.py install
}

