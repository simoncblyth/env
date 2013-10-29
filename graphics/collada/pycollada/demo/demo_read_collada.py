#!/usr/bin/env python
"""
In [1]: import collada as co

In [2]: dae = co.Collada('/tmp/test.dae')

In [3]: dae
Out[3]: <Collada geometries=1>

In [4]: dae.geometries
Out[4]: [<Geometry id=geometry0, 1 primitives>]

In [5]: dae.geometries[0]
Out[5]: <Geometry id=geometry0, 1 primitives>

In [6]: geom = dae.geometries[0]

In [7]: geom.
geom.bind               geom.createLineSet      geom.createPolylist     geom.double_sided       geom.load               geom.primitives         geom.sourceById         
geom.collada            geom.createPolygons     geom.createTriangleSet  geom.id                 geom.name               geom.save               geom.xmlnode            

In [7]: geom.double_sided
Out[7]: False

In [8]: geom.name
Out[8]: 'mycube'

In [9]: geom.bind?   
Type:       instancemethod
Base Class: <type 'instancemethod'>
String Form:<bound method Geometry.bind of <Geometry id=geometry0, 1 primitives>>
Namespace:  Interactive
File:       /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/pycollada-0.4-py2.6.egg/collada/geometry.py
Definition: geom.bind(self, matrix, materialnodebysymbol)
Docstring:
Binds this geometry to a transform matrix and material mapping.
The geometry's points get transformed by the given matrix and its
inputs get mapped to the given materials.

:param numpy.array matrix:
  A 4x4 numpy float matrix
  :param dict materialnodebysymbol:
    A dictionary with the material symbols inside the primitive
      assigned to :class:`collada.scene.MaterialNode` defined in the
        scene

        :rtype: :class:`collada.geometry.BoundGeometry`

In [10]: geom.primitives
Out[10]: [<TriangleSet length=12>]

In [11]: triset = geom.primitives[0]

In [12]: triset
Out[12]: <TriangleSet length=12>

In [13]: trilist = list(triset)

In [14]: len(trilist)
Out[14]: 12

In [15]: trilist
Out[15]: 
[<Triangle ([-50.  50.  50.], [-50. -50.  50.], [ 50. -50.  50.], "materialref")>,
 <Triangle ([-50.  50.  50.], [ 50. -50.  50.], [ 50.  50.  50.], "materialref")>,
  <Triangle ([-50.  50.  50.], [ 50.  50.  50.], [ 50.  50. -50.], "materialref")>,
   <Triangle ([-50.  50.  50.], [ 50.  50. -50.], [-50.  50. -50.], "materialref")>,
    <Triangle ([-50. -50. -50.], [ 50. -50. -50.], [ 50. -50.  50.], "materialref")>,
     <Triangle ([-50. -50. -50.], [ 50. -50.  50.], [-50. -50.  50.], "materialref")>,
      <Triangle ([-50.  50.  50.], [-50.  50. -50.], [-50. -50. -50.], "materialref")>,
       <Triangle ([-50.  50.  50.], [-50. -50. -50.], [-50. -50.  50.], "materialref")>,
        <Triangle ([ 50. -50.  50.], [ 50. -50. -50.], [ 50.  50. -50.], "materialref")>,
         <Triangle ([ 50. -50.  50.], [ 50.  50. -50.], [ 50.  50.  50.], "materialref")>,
          <Triangle ([ 50.  50. -50.], [ 50. -50. -50.], [-50. -50. -50.], "materialref")>,
           <Triangle ([ 50.  50. -50.], [-50. -50. -50.], [-50.  50. -50.], "materialref")>]



"""
