#!/usr/bin/env python
"""
DID NOT PURSUE THIS APPROACH, AS TOO INDIRECT GOING VIA GDML

"""
import os, logging
log = logging.getLogger(__name__)
import collada as co
import numpy as np


class DAE(co.Collada):
    def make_effect(self, id):
        """
        ::

            co.material.Effect(self, id, params, shadingtype, 
                    bumpmap=None, 
                    double_sided=False, 
                    emission=(0.0, 0.0, 0.0, 1.0), 
                    ambient=(0.0, 0.0, 0.0, 1.0), 
                    diffuse=(0.0, 0.0, 0.0, 1.0), 
                    specular=(0.0, 0.0, 0.0, 1.0), 
                    shininess=0.0, 
                    reflective=(0.0, 0.0, 0.0, 1.0), 
                    reflectivity=0.0, 
                    transparent=(0.0, 0.0, 0.0, 1.0), 
                    transparency=None, 
                    index_of_refraction=None, 
                    opaque_mode=None, 
                    xmlnode=None)
         """
        eff = co.material.Effect( id, [], "phong", diffuse=(1,0,0), specular=(0,1,0))
        return eff

    def add_geometry(self, id, name, materialref=None):
        """
        #. Dummy geometry
        #. triset wants a materialref ? would need to examine the GDML structure to associate material with geomety ? can we stay unbound until later ?

        """
        vert_floats = [-50,50,50,50,50,50,-50,-50,50,50,-50,50,-50,50,-50,50,50,-50,-50,-50,-50,50,-50,-50]
        norm_floats = [0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,1,0,0,-1,0,0,-1,0,0,-1,0,0,-1,0,-1,0,0,-1,0,0,-1,0,0,-1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,-1, 0,0,-1,0,0,-1,0,0,-1]
        vert_src = co.source.FloatSource( id+"-vert", np.array(vert_floats), ('X', 'Y', 'Z'))
        norm_src = co.source.FloatSource( id+"-norm", np.array(norm_floats), ('X', 'Y', 'Z'))
        geom = co.geometry.Geometry(self, id, name, [vert_src, norm_src]) 
        input_list = co.source.InputList()
        input_list.addInput(0, 'VERTEX', "#%s-vert" % id )
        input_list.addInput(1, 'NORMAL', "#%s-norm" % id )
        indices = np.array([0,0,2,1,3,2,0,0,3,2,1,3,0,4,1,5,5,6,0,4,5,6,4,7,6,8,7,9,3,10,6,8,3,10,2,11,0,12,4,13,6,14,0,12,6,14,2,15,3,16,7,17,5,18,3,16,5,18,1,19,5,20,7,21,6,22,5,20,6,22,4,23])
        triset = geom.createTriangleSet(indices, input_list, materialref )
        geom.primitives.append(triset)
        self.geometries.append(geom)

    def make_material(self, id, name):
        eff = self.make_effect(name)
        mat = co.material.Material( id, name, eff )
        return mat

    def add_material(self, id, name):
        mat = self.make_material(id, name)
        self.effects.append(mat.effect)
        self.materials.append(mat)

    def __str__(self):
        sio = StringIO()
        self.write(sio)
        return sio.getvalue()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dae = DAE()
    from gdml import GDML
    gd = GDML("$LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml")

    for vol in gd.walk():
        materialref, solidref, pvs = vol
        material = gd.material[materialref]
        solid = gd.solid[solidref]
        for pv in pvs:
            lv, pos, rot = gd.physvol[pv]
            



    

