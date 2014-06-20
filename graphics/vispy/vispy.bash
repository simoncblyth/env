# === func-gen- : graphics/vispy/vispy fgp graphics/vispy/vispy.bash fgn vispy fgh graphics/vispy
vispy-src(){      echo graphics/vispy/vispy.bash ; }
vispy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vispy-src)} ; }
vispy-vi(){       vi $(vispy-source) ; }
vispy-env(){      elocal- ; }
vispy-usage(){ cat << EOU

VISPY
=====

* https://github.com/vispy/vispy

A successor to glumpy ? 
Collaboration between authors of PyQtGraph, Visvis, Glumpy and Galry.

Vispy is an OpenGL-based interactive visualization library in Python. Its goal
is to make it easy to create beautiful and fast dynamic visualizations. For
example, scientific plotting of tens of millions of points, interacting with
complex polygonial models, and (dynamic) volume rendering. All thanks to the
graphics card’s hardware acceleration.


OpenGL ES alternatives to glMultiDrawArrays
---------------------------------------------------------------

* https://github.com/vispy/vispy/wiki/Trash.-Agenda

Lots of primitives in a single GL call without glMultiDrawArrays ?

* Multiple lines: GL_LINES + index buffer and we're fine: no copy during backing.
* Multiple lines: GL_LINES + double the number of points (+ extra point at the end of every line?)
* (Multiple lines: two GL_LINES calls without index buffer (+ extra point)?)

* For GL_LINE_STRIP, maybe a "discard" in the fragment shader, 
  with a varying variable specifying whether we are between two primitives (no integer). 
  Or alpha between primitives.

* For GL_TRIANGLE_STRIP, use degenerate triangles between primitives.
  With tick lines (GL_TRIANGLES and index buffer), no problem.

Backends
---------

::

    delta:vispy blyth$ l vispy/app/backends/
    total 208
    -rw-r--r--  1 blyth  staff   1415 Jun 20 17:31 __init__.py
    -rw-r--r--  1 blyth  staff  15199 Jun 20 17:31 _glfw.py
    -rw-r--r--  1 blyth  staff  15833 Jun 20 17:31 _glut.py
    -rw-r--r--  1 blyth  staff  14681 Jun 20 17:31 _pyglet.py
    -rw-r--r--  1 blyth  staff    924 Jun 20 17:31 _pyqt4.py
    -rw-r--r--  1 blyth  staff    940 Jun 20 17:31 _pyside.py
    -rw-r--r--  1 blyth  staff  13013 Jun 20 17:31 _qt.py
    -rw-r--r--  1 blyth  staff  14819 Jun 20 17:31 _sdl2.py
    -rw-r--r--  1 blyth  staff   5821 Jun 20 17:31 _template.py
    -rw-r--r--  1 blyth  staff    236 Jun 20 17:31 _test.py



EOU
}
vispy-dir(){ echo $(local-base)/env/graphics/vispy/vispy  ; }
vispy-cd(){  cd $(vispy-dir); }
vispy-mate(){ mate $(vispy-dir) ; }
vispy-get(){
   local dir=$(dirname $(vispy-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/vispy/vispy.git

}
