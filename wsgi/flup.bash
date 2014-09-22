# === func-gen- : wsgi/flup fgp wsgi/flup.bash fgn flup fgh wsgi
flup-src(){      echo wsgi/flup.bash ; }
flup-source(){   echo ${BASH_SOURCE:-$(env-home)/$(flup-src)} ; }
flup-vi(){       vi $(flup-source) ; }
flup-env(){      elocal- ; }
flup-usage(){ cat << EOU

FLUP : collection of WSGI servers
===================================

* https://pypi.python.org/pypi/flup/1.0

Installs
---------

D : daeserver vpython
~~~~~~~~~~~~~~~~~~~~~~~

::

    (daeserver_env)delta:env blyth$ flup-install
    /usr/local/env/geant4/geometry/daeserver_env/bin/easy_install
    ...
    Installed /usr/local/env/geant4/geometry/daeserver_env/lib/python2.7/site-packages/flup-1.0.3.dev_20110405-py2.7.egg
    ...








EOU
}
flup-dir(){ echo $(local-base)/env/wsgi/wsgi-flup ; }
flup-cd(){  cd $(flup-dir); }
flup-mate(){ mate $(flup-dir) ; }
flup-get(){
   local dir=$(dirname $(flup-dir)) &&  mkdir -p $dir && cd $dir




}

flup-install(){
   [ -z "$VIRTUAL_ENV" ] && echo this is intended for use with virtualenv && return
   which easy_install
   easy_install flup
}


