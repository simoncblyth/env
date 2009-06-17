# === func-gen- : python/virtualenv.bash fgp python/virtualenv.bash fgn virtualenv
virtualenv-src(){      echo python/virtualenv.bash ; }
virtualenv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(virtualenv-src)} ; }
virtualenv-vi(){       vi $(virtualenv-source) ; }
virtualenv-env(){      elocal- ; }
virtualenv-usage(){
  cat << EOU
     virtualenv-src : $(virtualenv-src)

EOU
}

virtualenv-get(){ sudo easy_install virtualenv ; }
virtualenv-version(){ virtualenv --version ; }
