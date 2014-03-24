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



