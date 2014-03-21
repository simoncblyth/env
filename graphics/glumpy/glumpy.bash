# === func-gen- : graphics/glumpy/glumpy fgp graphics/glumpy/glumpy.bash fgn glumpy fgh graphics/glumpy
glumpy-src(){      echo graphics/glumpy/glumpy.bash ; }
glumpy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(glumpy-src)} ; }
glumpy-vi(){       vi $(glumpy-source) ; }
glumpy-env(){      elocal- ; }
glumpy-usage(){ cat << EOU

GLUMPY
=======

* https://code.google.com/p/glumpy/
* http://groups.google.com/group/glumpy-users

glumpy is a small python library for the rapid vizualization of numpy arrays,
(mainly two dimensional) that has been designed with efficiency in mind. If you
want to draw nice figures for inclusion in a scientific article, you’d better
use matplotlib. If you want to have a sense of what’s going on in your
simulation while it is running, then maybe glumpy can help you.

glumpy is made on top of PyOpenGL (http://pyopengl.sourceforge.net/) and since
glumpy is dedicated to numpy visualization, you obviously need numpy
(http://numpy.scipy.org/). You will also need IPython
(http://ipython.scipy.org/) for running interactive sessions where you can
interact with glumpy.

Some demos require matplotlib (http://matplotlib.sourceforge.net/) and scipy
(http://www.scipy.org/) as well but this is optional.

installations
--------------

Delta
~~~~~

Into chroma virtualenv python (based on macports python 2.7.6)::

    python setup.py install
    ...
    Writing /usr/local/env/chroma_env/lib/python2.7/site-packages/glumpy-0.2.1-py2.7.egg-info


demos
------

obj-viewer
~~~~~~~~~~~

Nice trackball operation 

/usr/local/env/graphics/glumpy/glumpy/demos/obj-viewer.py






EOU
}
glumpy-dir(){ echo $(local-base)/env/graphics/glumpy/glumpy ; }
glumpy-cd(){  cd $(glumpy-dir); }
glumpy-mate(){ mate $(glumpy-dir) ; }
glumpy-get(){
   local dir=$(dirname $(glumpy-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://code.google.com/p/glumpy/


}
