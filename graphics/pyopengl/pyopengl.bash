# === func-gen- : graphics/pyopengl/pyopengl fgp graphics/pyopengl/pyopengl.bash fgn pyopengl fgh graphics/pyopengl
pyopengl-src(){      echo graphics/pyopengl/pyopengl.bash ; }
pyopengl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pyopengl-src)} ; }
pyopengl-vi(){       vi $(pyopengl-source) ; }
pyopengl-env(){      elocal- ; }
pyopengl-usage(){ cat << EOU

PYOPENGL
========

* http://pyopengl.sourceforge.net

PyOpenGL is the most common cross platform Python binding to OpenGL and related
APIs. The binding is created using the standard ctypes library, and is provided
under an extremely liberal BSD-style Open-Source license.

PyOpenGL 3.x supports OpenGL v1.1+

References
----------

* http://cyrille.rossant.net/tag/pyopengl/


PIP Package names
------------------

::

    PyOpenGL
    PyOpenGL_accelerate
    PyOpenGL Demo
    OpenGLContext

Versions
---------

* https://pypi.python.org/pypi/PyOpenGL

Current pip versions::

    PyOpenGL 3.1.0b1   2014-02-11   
    PyOpenGL 3.0.2     2012-10-02
    PyOpenGL 3.0.1     2010-02-25

Macports py27 versions::

    > port search py27-opengl

    py27-opengl @3.0.2_1 (python, graphics)
        Python binding to OpenGL

    py27-opengl-accelerate @3.0.2_2 (python, graphics)
        Acceleration code for PyOpenGL

    Found 2 ports.


Installation
-------------

* http://pyopengl.sourceforge.net/documentation/installation.html

pip is recommended.

G4PB: macports py26
~~~~~~~~~~~~~~~~~~~~~~

Macports install attempts and fails to upgrade python for py26-opengl and py26-pip::

    g4pb:~ blyth$ sudo port install py26-opengl
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    --->  Computing dependencies for python26
    --->  Verifying checksums for python26
    Error: Checksum (md5) mismatch for Python-2.6.9.tar.xz
    Error: Checksum (rmd160) mismatch for Python-2.6.9.tar.xz
    Error: Checksum (sha256) mismatch for Python-2.6.9.tar.xz
    Error: org.macports.checksum for port python26 returned: Unable to verify file checksums
    Please see the log file for port python26 for details:
        /opt/local/var/macports/logs/_opt_local_var_macports_sources_rsync.macports.org_release_ports_lang_python26/python26/main.log
    Error: Unable to upgrade port: 1
    Error: Unable to execute port: upgrade py26-setuptools failed
    g4pb:~ blyth$ 

::

    g4pb:~ blyth$ sudo port install py26-pip
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    --->  Computing dependencies for python26
    --->  Verifying checksums for python26
    Error: Checksum (md5) mismatch for Python-2.6.9.tar.xz
    Error: Checksum (rmd160) mismatch for Python-2.6.9.tar.xz
    Error: Checksum (sha256) mismatch for Python-2.6.9.tar.xz
    Error: org.macports.checksum for port python26 returned: Unable to verify file checksums
    Please see the log file for port python26 for details:
        /opt/local/var/macports/logs/_opt_local_var_macports_sources_rsync.macports.org_release_ports_lang_python26/python26/main.log
    Error: Unable to upgrade port: 1
    Error: Unable to execute port: upgrade python26 failed
    g4pb:~ blyth$ 


Ah I already have macports pip for py26::

    g4pb:~ blyth$ which pip-2.6
    /opt/local/bin/pip-2.6

Huh, also already have pyopengl::

    g4pb:~ blyth$ pip-2.6 install PyOpenGL PyOpenGL_accelerate
    Requirement already satisfied (use --upgrade to upgrade): PyOpenGL in /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages
    Requirement already satisfied (use --upgrade to upgrade): PyOpenGL-accelerate in /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages
    Cleaning up...




Delta: chroma virtual python (based on macports py 2.7.6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Surprised to get the older version::

    (chroma_env)delta:collada blyth$ pip install PyOpenGL PyOpenGL_accelerate
    Downloading/unpacking PyOpenGL
      Downloading PyOpenGL-3.0.2.tar.gz (891kB): 891kB downloaded
    ...
    Downloading/unpacking PyOpenGL-accelerate
      Downloading PyOpenGL-accelerate-3.0.2.tar.gz (235kB): 235kB downloaded
      Running setup.py egg_info for package PyOpenGL-accelerate
        
    Installing collected packages: PyOpenGL, PyOpenGL-accelerate
      Running setup.py install for PyOpenGL


Demo::

    (chroma_env)delta:pyopengl_examples blyth$ pip install PyOpenGL-Demo
    Downloading/unpacking PyOpenGL-Demo
      Downloading PyOpenGL-Demo-3.0.0.tar.gz (1.3MB):  75%  991kB




EOU
}
pyopengl-dir(){ echo $(local-base)/env/graphics/pyopengl/graphics/pyopengl-pyopengl ; }
pyopengl-cd(){  cd $(pyopengl-dir); }
pyopengl-mate(){ mate $(pyopengl-dir) ; }
pyopengl-get(){
   local dir=$(dirname $(pyopengl-dir)) &&  mkdir -p $dir && cd $dir

}

pyopengl-demo-dir(){ echo $VIRTUAL_ENV/lib/python2.7/site-packages/PyOpenGL-Demo ; }
pyopengl-demo-cd(){  cd $(pyopengl-demo-dir) ; }



pyopengl-version(){
  python -c "import OpenGL.version ; print OpenGL.version.__version__"
}



