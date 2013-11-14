PyCOLLADA Triangulation
========================


It does triangulate from a polylist. What about the other higher level objects.

::

    In [12]: geom
    Out[12]: <Geometry id=box, 1 primitives>

    In [15]: prim = geom.primitives[0]

    In [16]: prim.__class__
    Out[16]: collada.polygons.Polygons

    In [17]: prim.
    prim.bind                  prim.load                  prim.maxvertexindex        prim.npolygons             prim.polystarts            prim.texbinormalset        prim.textangentset         prim.vertex_index
    prim.getInputList          prim.material              prim.nindices              prim.nvertices             prim.save                  prim.texcoord_indexset     prim.triangleset           prim.xmlnode
    prim.index                 prim.maxnormalindex        prim.normal                prim.polyends              prim.sources               prim.texcoordset           prim.vcounts               
    prim.indices               prim.maxtexcoordsetindex   prim.normal_index          prim.polyindex             prim.texbinormal_indexset  prim.textangent_indexset   prim.vertex                

    In [18]: prim.triangleset?
    Definition: prim.triangleset(self)
    Docstring:
    This performs a simple triangulation of the polylist using the fanning method.

    :rtype: :class:`collada.triangleset.TriangleSet`

    In [19]: prim.triangleset()
    Out[19]: <TriangleSet length=12>

    In [20]: tris = prim.triangleset()

    In [21]: tris[0]
    Out[21]: <Triangle ([-0.5  0.5  0.5], [-0.5 -0.5  0.5], [ 0.5 -0.5  0.5], "WHITE")>

    In [22]: tris[1]
    Out[22]: <Triangle ([-0.5  0.5  0.5], [ 0.5 -0.5  0.5], [ 0.5  0.5  0.5], "WHITE")>

    In [23]: tris[-1]
    Out[23]: <Triangle ([ 0.5  0.5 -0.5], [-0.5 -0.5 -0.5], [-0.5  0.5 -0.5], "WHITE")>


