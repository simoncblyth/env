# === func-gen- : python/pip fgp python/pip.bash fgn pip fgh python
pip-src(){      echo python/pip.bash ; }
pip-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pip-src)} ; }
pip-vi(){       vi $(pip-source) ; }
pip-env(){      elocal- ; }
pip-usage(){ cat << EOU

PIP Installs Packages
======================

* http://www.pip-installer.org/en/latest/configuration.html



EOU
}
pip-dir(){ echo $(local-base)/env/python/python-pip ; }
pip-cd(){  cd $(pip-dir); }
pip-mate(){ mate $(pip-dir) ; }
pip-get(){
   local dir=$(dirname $(pip-dir)) &&  mkdir -p $dir && cd $dir

}


pip-log(){
   vi ~/.pip/pip.log
}
