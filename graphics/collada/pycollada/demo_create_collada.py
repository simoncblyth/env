#!/usr/bin/env python
"""
http://pycollada.github.io/creating.html

"""

import collada as co, numpy as np
from StringIO import StringIO

dae = co.Collada()
effect = co.material.Effect("effect0", [], "phong", diffuse=(1,0,0), specular=(0,1,0))
mat = co.material.Material("material0", "mymaterial", effect)

dae.effects.append(effect)
dae.materials.append(mat)

vert_floats = [-50,50,50,50,50,50,-50,-50,50,50,-50,50,-50,50,-50,50,50,-50,-50,-50,-50,50,-50,-50]
normal_floats = [0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,1,0,0,-1,0,0,-1,0,0,-1,0,0,-1,0,-1,0,0,-1,0,0,-1,0,0,-1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,-1, 0,0,-1,0,0,-1,0,0,-1]

vert_src = co.source.FloatSource("cubeverts-array", np.array(vert_floats), ('X', 'Y', 'Z'))
normal_src = co.source.FloatSource("cubenormals-array", np.array(normal_floats), ('X', 'Y', 'Z'))
geom = co.geometry.Geometry(dae, "geometry0", "mycube", [vert_src, normal_src])

input_list = co.source.InputList()
input_list.addInput(0, 'VERTEX', "#cubeverts-array")
input_list.addInput(1, 'NORMAL', "#cubenormals-array")
indices = np.array([0,0,2,1,3,2,0,0,3,2,1,3,0,4,1,5,5,6,0,4,5,6,4,7,6,8,7,9,3,10,6,8,3,10,2,11,0,12,4,13,6,14,0,12,6,14,2,15,3,16,7,17,5,18,3,16,5,18,1,19,5,20,7,21,6,22,5,20,6,22,4,23])

triset = geom.createTriangleSet(indices, input_list, "materialref")

geom.primitives.append(triset)
dae.geometries.append(geom)

matnode = co.scene.MaterialNode("materialref", mat, inputs=[])
geonode = co.scene.GeometryNode(geom, [matnode])
node = co.scene.Node("node0", children=[geonode])
myscene = co.scene.Scene("myscene", [node])

dae.scenes.append(myscene)
dae.scene = myscene

out = StringIO()
dae.write(out)


print out.getvalue()


