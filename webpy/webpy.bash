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

D : daeserver vpython
~~~~~~~~~~~~~~~~~~~~~~~~

::
  
    daeserver-
    daeserver--
    which easy_install 
    easy_install web.py

    daeserver_env)delta:~ blyth$ easy_install web.py 
    Searching for web.py
    Reading https://pypi.python.org/simple/web.py/
    Best match: web.py 0.37
    Downloading https://pypi.python.org/packages/source/w/web.py/web.py-0.37.tar.gz#md5=93375e3f03e74d6bf5c5096a4962a8db
    Processing web.py-0.37.tar.gz
    Writing /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BFmYwY/web.py-0.37/setup.cfg
    Running web.py-0.37/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BFmYwY/web.py-0.37/egg-dist-tmp-wV9b_D
    zip_safe flag not set; analyzing archive contents...
    web.application: module references __file__
    web.debugerror: module references __file__
    Adding web.py 0.37 to easy-install.pth file

    Installed /usr/local/env/geant4/geometry/daeserver_env/lib/python2.7/site-packages/web.py-0.37-py2.7.egg
    Processing dependencies for web.py
    Finished processing dependencies for web.py
    (daeserver_env)delta:~ blyth$ 



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
webpy-dir(){ echo $(env-home)/webpy ; }
webpy-cd(){  cd $(webpy-dir); }

webpy-install(){
   [ -z "$VIRTUAL_ENV" ] && echo $msg this is intended to be used with virtualenv see daeserver-vi as example && return 
   which easy_install
   easy_install web.py 
}

webpy-tutorial-server(){
   g4daeserver- 
   $(g4daeserver-vdir)/bin/python $(webpy-dir)/tutorial/code.py 
}

webpy-tutorial-test(){
   curl http://0.0.0.0:8080/
}



