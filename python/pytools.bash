# === func-gen- : python/pytools fgp python/pytools.bash fgn pytools fgh python
pytools-src(){      echo python/pytools.bash ; }
pytools-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pytools-src)} ; }
pytools-vi(){       vi $(pytools-source) ; }
pytools-env(){      elocal- ; }
pytools-usage(){ cat << EOU

PYTOOLS
=========

Miscellaneous Python lifesavers, from PyCUDA author Andreas Kloeckner

* https://github.com/inducer/pytools






EOU
}
pytools-dir(){ echo $(local-base)/env/python/python-pytools ; }
pytools-cd(){  cd $(pytools-dir); }
pytools-mate(){ mate $(pytools-dir) ; }
pytools-get(){
   local dir=$(dirname $(pytools-dir)) &&  mkdir -p $dir && cd $dir

}
