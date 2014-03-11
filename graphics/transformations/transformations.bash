# === func-gen- : graphics/transformations/transformations fgp graphics/transformations/transformations.bash fgn transformations fgh graphics/transformations
transformations-src(){      echo graphics/transformations/transformations.bash ; }
transformations-source(){   echo ${BASH_SOURCE:-$(env-home)/$(transformations-src)} ; }
transformations-vi(){       vi $(transformations-source) ; }
transformations-env(){      elocal- ; }
transformations-usage(){ cat << EOU

TRANSFORMATIONS
================

BSD licensed set of utilities in python and C

* http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
* http://www.lfd.uci.edu/~gohlke/code/transformations.c.html

INSTALLATION
------------

Prior to compilation of the C extension the python import gives warning, but functionality 
is still provided by the pure python implementation::

    In [18]: from env.graphics.transformations.transformations import identity_matrix
    /usr/local/env/chroma_env/lib/python2.7/site-packages/env/graphics/transformations/transformations.py:1888: UserWarning: failed to import module _transformations
      warnings.warn("failed to import module %s" % name)

Arcball
--------

Imagine a virtual ball centered just behind the screen with defined pixel center and radius.
Points on the screen can easily be mapped to a position on the virtual ball 

Arcball holds a quaternion (representing an orientation) 
which is updated on the basis of mouse drags as if they manipulate 
the virtual ball.

* http://en.wikibooks.org/wiki/OpenGL_Programming/Modern_OpenGL_Tutorial_Arcball

::

    In [38]: arcball_map_to_sphere( point=(0,100), center=(200,100), radius=100 )
    Out[38]: array([-1.,  0.,  0.])

    In [39]: arcball_map_to_sphere( point=(200,200), center=(200,100), radius=100 )
    Out[39]: array([ 0., -1.,  0.])

    In [40]: arcball_map_to_sphere( point=(200,-200), center=(200,100), radius=100 )
    Out[40]: array([ 0.,  1.,  0.])

    In [41]: arcball_map_to_sphere( point=(200,100), center=(200,100), radius=100 )
    Out[41]: array([ 0.,  0.,  1.])

    In [42]: arcball_map_to_sphere( point=(250,100), center=(200,100), radius=100 )
    Out[42]: array([ 0.5      ,  0.       ,  0.8660254])


Axis angle rep
~~~~~~~~~~~~~~~

Chroma performs rotations using axis and angle, how to get that from the 
arcball quaternion ?

* http://en.wikipedia.org/wiki/Axis–angle_representation
* http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/

The below looks invertible::

    1231 def quaternion_about_axis(angle, axis):
    1232     """Return quaternion for rotation about axis.
    1233 
    1234     >>> q = quaternion_about_axis(0.123, [1, 0, 0])
    1235     >>> numpy.allclose(q, [0.99810947, 0.06146124, 0, 0])
    1236     True
    1237 
    1238     """
    1239     q = numpy.array([0.0, axis[0], axis[1], axis[2]])
    1240     qlen = vector_norm(q)
    1241     if qlen > _EPS:
    1242         q *= math.sin(angle/2.0) / qlen
    1243     q[0] = math.cos(angle/2.0)
    1244     return q











EOU
}
transformations-dir(){ echo $(env-home)/graphics/transformations ; }
transformations-cd(){  cd $(transformations-dir); }
transformations-mate(){ mate $(transformations-dir) ; }
transformations-get(){
   local dir=$(transformations-dir) &&  mkdir -p $dir && cd $dir

   curl -L -O http://www.lfd.uci.edu/~gohlke/code/transformations.py
   curl -L -O http://www.lfd.uci.edu/~gohlke/code/transformations.c

}
