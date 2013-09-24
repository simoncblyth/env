# === func-gen- : tools/pyrtf fgp tools/pyrtf.bash fgn pyrtf fgh tools
pyrtf-src(){      echo tools/pyrtf.bash ; }
pyrtf-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pyrtf-src)} ; }
pyrtf-vi(){       vi $(pyrtf-source) ; }
pyrtf-env(){      elocal- ; }
pyrtf-usage(){ cat << EOU

PYRTF : Parse and write RTF
============================

* https://launchpad.net/pyrtf
* http://code.google.com/p/pyrtf-ng/

The total lack of any usage documentation and typos in the code
makes me think this project is dead.

installs
----------

* into python 2.5.6 macports


itextrtf
--------

Yuck sourceforge

* http://sourceforge.net/projects/itextrtf/
* http://itextrtf.sourceforge.net/


EOU
}
pyrtf-dir(){ echo $(local-base)/env/tools/pyrtf ; }
pyrtf-cd(){  cd $(pyrtf-dir); }
pyrtf-mate(){ mate $(pyrtf-dir) ; }
pyrtf-get(){
   local dir=$(dirname $(pyrtf-dir)) &&  mkdir -p $dir && cd $dir

   bzr branch lp:pyrtf

}

pyrtf-build(){
   pyrtf-cd

   python setup.py build
   sudo python setup.py install


}

pyrtf-check(){  
   python -c "import rtfng"
}

