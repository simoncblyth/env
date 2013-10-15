# === func-gen- : webpy/webpy fgp webpy/webpy.bash fgn webpy fgh webpy
webpy-src(){      echo webpy/webpy.bash ; }
webpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(webpy-src)} ; }
webpy-vi(){       vi $(webpy-source) ; }
webpy-env(){      elocal- ; }
webpy-usage(){ cat << EOU

WEBPY
======

Lightweight web server.

* http://webpy.org/docs/0.3/tutorial
* http://webpy.org/src/

G
---

::

    sudo port install py26-webpy




EOU
}
webpy-dir(){ echo $(local-base)/env/webpy/webpy-webpy ; }
webpy-cd(){  cd $(webpy-dir); }
webpy-mate(){ mate $(webpy-dir) ; }
webpy-get(){
   local dir=$(dirname $(webpy-dir)) &&  mkdir -p $dir && cd $dir

}
