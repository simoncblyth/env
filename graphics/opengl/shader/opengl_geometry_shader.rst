OpenGL Geometry Shader
=======================

New API
-----------

* http://www.opengl.org/wiki/Geometry_Shader
* http://open.gl/geometry

Describes the new Geometry Shader (that are not yet available to me)

Since OpenGL 3.2 there is a third optional type of shader that sits between the
vertex and fragment shaders, known as the geometry shader. 


Issue with changing shader
----------------------------

Swapping out a geometry shader works, swapping in does not  

#. Mesa (OpenGL in software) implementation of geometry shader might be illuminating 

   * http://lists.freedesktop.org/archives/mesa-dev/2012-July/024792.html
  

Debugging Geometry Shaders
---------------------------

* http://stackoverflow.com/questions/8795856/glsl-1-5-simple-geometry-shader


Geometry Shader Details
------------------------

* http://www.informit.com/articles/article.aspx?p=2120983&seqNum=2


Older API
-----------

* http://www.opengl.org/wiki/Geometry_Shader_Examples

* removed from core OpenGL 3.1 and above (they are only deprecated in OpenGL 3.0). 
* It is recommended that you not use this functionality in your programs.

  * **BUT: THIS IS ONLY WAY TO USE GEOMETRY SHADERS CURRENTLY**

::

    //GEOMETRY SHADER
     #version 120
     #extension GL_ARB_geometry_shader4 : enable
     ///////////////////////
     void main()
     {
       //increment variable
       int i;
       vec4 vertex;
       /////////////////////////////////////////////////////////////
       //This example has two parts
       //   step a) draw the primitive pushed down the pipeline
       //            there are gl_VerticesIn # of vertices
       //            put the vertex value into gl_Position
       //            use EmitVertex => 'create' a new vertex
       //           use EndPrimitive to signal that you are done creating a primitive!
       //   step b) create a new piece of geometry
       //           I just do the same loop, but I negate the vertex.z
       //   result => the primitive is now mirrored.
       //
       // Pass-thru!
       //
       for(i = 0; i < gl_VerticesIn; i++)
       {
         gl_Position = gl_PositionIn[i];
         EmitVertex();
       }
       EndPrimitive();

       // New piece of geometry!
       for(i = 0; i < gl_VerticesIn; i++)
       {
         vertex = gl_PositionIn[i];
         vertex.z = -vertex.z;
         gl_Position = vertex;
         EmitVertex();
       }
       EndPrimitive();
     }




OSX
-----

* https://developer.apple.com/library/mac/documentation/graphicsimaging/conceptual/opengl-macprogguide/opengl_shaders/opengl_shaders.html
* http://www.opengl.org/registry/specs/EXT/geometry_shader4.txt
* http://www.opengl.org/registry/specs/ARB/geometry_shader4.txt


Examples
----------

* http://stackoverflow.com/questions/3936368/geometry-shader-doesnt-do-anything-when-fed-gl-points

  * got this to work, by changing `distance` to `mydistance`

* http://programminglinuxgames.blogspot.tw/2011/03/pyopengl-geometry-shaders-python-and.html
* http://enja.org/2011/05/01/can-you-see-the-force/
* http://blog.mikepan.com








