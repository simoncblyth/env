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
* http://johnpaulett.com/2008/09/20/getting-restful-with-webpy/


Requirements:

* flup
* http://www.web2pyslices.com/slice/show/1466/nginx-and-scgi


Accepting PUT
---------------





INSTALLS
----------

N
~~~

source py2.5.1::

    python- source
    easy_install web.py


Needed a flup install for running daeserver.py::

  File "/data1/env/system/python/Python-2.5.1/lib/python2.5/site-packages/web.py-0.37-py2.5.egg/web/wsgi.py", line 21, in runscgi
    import flup.server.scgi as flups
  ImportError: No module named flup.server.scgi

   easy_install flup

G
~~~

macports py26::

    sudo port install py26-webpy



EOU
}
webpy-dir(){ echo $(local-base)/env/webpy/webpy-webpy ; }
webpy-cd(){  cd $(webpy-dir); }
webpy-mate(){ mate $(webpy-dir) ; }
webpy-get(){
   local dir=$(dirname $(webpy-dir)) &&  mkdir -p $dir && cd $dir

}
