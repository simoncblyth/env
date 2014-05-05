# === func-gen- : chroma/chromaserver fgp chroma/chromaserver.bash fgn chromaserver fgh chroma
chromaserver-src(){      echo chroma/chromaserver.bash ; }
chromaserver-source(){   echo ${BASH_SOURCE:-$(env-home)/$(chromaserver-src)} ; }
chromaserver-vi(){       vi $(chromaserver-source) ; }
chromaserver-env(){      elocal- ; }
chromaserver-usage(){ cat << EOU

CHROMA SERVER
==============

https://github.com/mastbaum/chroma-server

This appears to now be integrated with chroma implemented
in chroma/bin/chroma-server


git issue from belle7
------------------------

Smth funny with belle7 or its network, I can clone it from elsewhere::

    blyth@belle7 chroma]$ git clone https://github.com/mastbaum/chroma-server
    Initialized empty Git repository in /data1/env/local/env/chroma/chroma-server/.git/
    Cannot get remote repository information.
    Perhaps git-update-server-info needs to be run there?

So scp from D::

    delta:chroma blyth$ scp -r chroma-server N:/data1/env/local/env/chroma/


ZMQ
----

::

    sudo port install zmq   # zeromq-3.2.3


install
----------

`setup.py` invoked make but there is no Makefile ? so `rm -rf /usr/local/env/chroma_env/lib/python2.7/site-packages/chroma_server`

::

    (chroma_env)delta:chroma-server blyth$ python setup.py install
    running install
    make: *** No targets specified and no makefile found.  Stop.
    running build
    running build_py
    creating build
    creating build/lib
    creating build/lib/chroma_server
    copying chroma_server/__init__.py -> build/lib/chroma_server
    copying chroma_server/photons.py -> build/lib/chroma_server
    copying chroma_server/serialize.py -> build/lib/chroma_server
    copying chroma_server/server.py -> build/lib/chroma_server
    copying chroma_server/serialize.C -> build/lib/chroma_server
    running build_scripts
    creating build/scripts-2.7
    copying and adjusting bin/chroma-server -> build/scripts-2.7
    changing mode of build/scripts-2.7/chroma-server from 644 to 755
    running install_lib
    creating /usr/local/env/chroma_env/lib/python2.7/site-packages/chroma_server
    copying build/lib/chroma_server/__init__.py -> /usr/local/env/chroma_env/lib/python2.7/site-packages/chroma_server
    copying build/lib/chroma_server/photons.py -> /usr/local/env/chroma_env/lib/python2.7/site-packages/chroma_server
    copying build/lib/chroma_server/serialize.C -> /usr/local/env/chroma_env/lib/python2.7/site-packages/chroma_server
    copying build/lib/chroma_server/serialize.py -> /usr/local/env/chroma_env/lib/python2.7/site-packages/chroma_server
    copying build/lib/chroma_server/server.py -> /usr/local/env/chroma_env/lib/python2.7/site-packages/chroma_server
    byte-compiling /usr/local/env/chroma_env/lib/python2.7/site-packages/chroma_server/__init__.py to __init__.pyc
    byte-compiling /usr/local/env/chroma_env/lib/python2.7/site-packages/chroma_server/photons.py to photons.pyc
    byte-compiling /usr/local/env/chroma_env/lib/python2.7/site-packages/chroma_server/serialize.py to serialize.pyc
    byte-compiling /usr/local/env/chroma_env/lib/python2.7/site-packages/chroma_server/server.py to server.pyc
    running install_scripts
    copying build/scripts-2.7/chroma-server -> /usr/local/env/chroma_env/bin
    changing mode of /usr/local/env/chroma_env/bin/chroma-server to 755
    running install_egg_info
    Writing /usr/local/env/chroma_env/lib/python2.7/site-packages/chroma_server-0.5-py2.7.egg-info
    (chroma_env)delta:chroma-server blyth$ 





EOU
}
chromaserver-dir(){ echo $(local-base)/env/chroma/chroma-server ; }
chromaserver-cd(){  cd $(chromaserver-dir); }
chromaserver-mate(){ mate $(chromaserver-dir) ; }
chromaserver-get(){
   local dir=$(dirname $(chromaserver-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/mastbaum/chroma-server

}
