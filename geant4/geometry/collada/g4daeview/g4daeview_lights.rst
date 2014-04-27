
G4DAEVIEW Lights
================

reference
-----------

* http://www.opengl.org/archives/resources/faq/technical/lights.htm

* http://www.talisman.org/opengl-1.1/Reference/glLight.html

The position is transformed by the modelview matrix when glLight is called
(just as if it were a point), and it is stored in eye coordinates. If the w
component of the position is 0, the light is treated as a directional source.
Diffuse and specular lighting calculations take the light's direction, but not
its actual position, into account, and attenuation is disabled. Otherwise,
diffuse and specular lighting calculations are based on the actual location of
the light in eye coordinates, and attenuation is enabled. The initial position
is (0, 0, 1, 0); thus, the initial light source is directional, parallel to,
and in the direction of the -Z axis.



Light Positioning
-------------------

* http://www.glprogramming.com/red/chapter05.html#name13

As shown, you supply a vector of four values (x, y, z, w) for the GL_POSITION
parameter. If the last value, w, is zero, the corresponding light source is a
directional one, and the (x, y, z) values describe its direction. This
direction is transformed by the modelview matrix. By default, GL_POSITION is
(0, 0, 1, 0), which defines a directional light that points along the negative
z-axis. (Note that nothing prevents you from creating a directional light with
the direction of (0, 0, 0), but such a light won't help you much.)

If the w value is nonzero, the light is positional, and the (x, y, z) values
specify the location of the light in homogeneous object coordinates. (See
Appendix F.) This location is transformed by the modelview matrix and stored in
eye coordinates. (See "Controlling a Light's Position and Direction" for more
information about how to control the transformation of the light's location.)
Also, by default, a positional light radiates in all directions, but you can
restrict it to producing a cone of illumination by defining the light as a
spotlight. (See "Spotlights" for an explanation of how to define a light as a
spotlight.)
i

Controlling a Light's Position and Direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Moving the Light Source Together with Your Viewpoint

* http://www.glprogramming.com/red/chapter05.html#name10

To create a light that moves along with the viewpoint, you need to set the
light position before the viewing transformation. Then the viewing
transformation affects both the light and the viewpoint in the same way.
Remember that the light position is stored in eye coordinates, and this is one
of the few times when eye coordinates are critical. In Example 5-7, the light
position is defined in init(), which stores the light position at (0, 0, 0) in
eye coordinates. In other words, the light is shining from the lens of the
camera.



Understanding Lights
---------------------

Moving the red light along z while looking down z::
    
              
                         r1     r2       r3    
              look       <      eye
                |--------|-------|-------|---------   Z
                0        1       2       3

::

    g4daeview.py -g 5000:6000 -t "" --eye="0,0,2" --look="0,0,0" --up="0,0.001,1" --lights="r" --rlight="0,0,1"  

        # only PMTs lit red, not background

    g4daeview.py -g 5000:6000 -t "" --eye="0,0,2" --look="0,0,0" --up="0,0.001,1" --lights="r" --rlight="0,0,2"  

        # light at eye position, PMTs and background lit red
        # perturbing z offset, makes background fade to grey around zoffset 0.4, 
        # light must be behind you to light the background  

    g4daeview.py -g 5000:6000 -t "" --eye="0,0,2" --look="0,0,0" --up="0,0.001,1" --lights="r" --rlight="0,0,3"

        # bright red background, fading to grey at zoffset 1.4

    g4daeview.py -g 5000:6000 -t "" --eye="0,0,2" --look="0,0,0" --up="0,0.001,1" --lights="r" --rlight="-1,1,2"


Original glumpy trackball constrained distance to be greater than 1::

    213         gl.glTranslate (self._x, self._y, -self._distance)
    ...
    238     def _set_distance(self, distance):
    239         self._distance = distance
    240         if self._distance < 1: self._distance= 1
    241     distance = property(_get_distance, _set_distance,
    242                         doc='''Scene distance from point of view''')
    243 


::

    g4daeview.py -g 5000:6000 -t "" --eye="0,0,2" --look="0,0,0" --up="0,0.001,1" --lights="rgb" --rlight="-1,1,1" --glight="1,1,1" --blight="0,-1,1" --flight 10


    g4daeview.py -g 5000:6000 --lights="rgb" --rlight="-1,1,-3" --glight="1,1,-3" --blight="0,-1,-3"   -t "" --eye="0,0.001,-3"  --wlight=0.
  



glumpy lights
---------------

::

    (chroma_env)delta:glumpy blyth$ find . -name '*.py' -exec grep -l Light {} \;
    ./demos/atb.py
    ./demos/colormaps.py   
    ./demos/histogram.py
    ./demos/obj-viewer.py
    ./glumpy/colormap.py                 # smth else
    ./glumpy/figure.py                   # figure on_init setup
    ./glumpy/graphics/vertex_buffer.py   # only the testing main
    (chroma_env)delta:glumpy blyth$ 


refs
-----

* http://www.glprogramming.com/red/chapter05.html#name5
* http://sjbaker.org/steve/omniv/opengl_lighting.html




