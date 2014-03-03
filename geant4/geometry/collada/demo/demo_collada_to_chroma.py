#!/usr/bin/env python
"""
::

   ipython demo_collada_to_chroma.py demo.dae -i 

"""
import os, sys, logging, re
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)   # chroma has weird logging, forcing this placement 

from env.geant4.geometry.collada.daenode import DAENode
from chroma.geometry import Mesh, Solid, Material, Surface, Geometry


matptn = re.compile("^__dd__Materials__(\S*)0x\S{7}$")
def matshorten(name):
    m = matptn.match(name)
    if m:
        return m.group(1) 
    return name
         


class ColladaToChroma(object):
    def __init__(self, dae):
        self.geo = Geometry(detector_material=None)    # bialkali ?
        self.dae = dae
        self.vcount = 0
        self.materials = {}

    def convert_materials(self):
        """
        Chroma materials default to None, 3 settings:

        * refractive_index
        * absorption_length
        * scattering_length

        And defaults to zero, 2 settings:

        * reemission_prob
        * reemission_cdf

        G4DAE materials have an extra attribute dict that 
        contains keys such as

        * RINDEX
        * ABSLENGTH
        * RAYLEIGH

        Uncertain of key correspondence, especially REEMISSIONPROB
        and what about reemission_cdf ? Possibly the many Scintillator
        keys can provide that ?

        Probably many the scintillator keys are only relevant to photon 
        production rather than photon propagation, so they are irrelevant 
        to Chroma.

        Which materials have each::

             EFFICIENCY                     [1 ] Bialkali 
             SLOWTIMECONSTANT               [2 ] GdDopedLS,LiquidScintillator 
             GammaFASTTIMECONSTANT          [2 ] GdDopedLS,LiquidScintillator 
             ReemissionSLOWTIMECONSTANT     [2 ] GdDopedLS,LiquidScintillator 
             REEMISSIONPROB                 [2 ] GdDopedLS,LiquidScintillator 
             AlphaFASTTIMECONSTANT          [2 ] GdDopedLS,LiquidScintillator 
             ReemissionFASTTIMECONSTANT     [2 ] GdDopedLS,LiquidScintillator 
             SLOWCOMPONENT                  [2 ] GdDopedLS,LiquidScintillator 
             YIELDRATIO                     [2 ] GdDopedLS,LiquidScintillator 
             FASTCOMPONENT                  [2 ] GdDopedLS,LiquidScintillator 
             NeutronFASTTIMECONSTANT        [2 ] GdDopedLS,LiquidScintillator 
             ReemissionYIELDRATIO           [2 ] GdDopedLS,LiquidScintillator 
             NeutronYIELDRATIO              [2 ] GdDopedLS,LiquidScintillator 
             GammaYIELDRATIO                [2 ] GdDopedLS,LiquidScintillator 
             SCINTILLATIONYIELD             [2 ] GdDopedLS,LiquidScintillator 
             AlphaYIELDRATIO                [2 ] GdDopedLS,LiquidScintillator 
             RESOLUTIONSCALE                [2 ] GdDopedLS,LiquidScintillator 
             GammaSLOWTIMECONSTANT          [2 ] GdDopedLS,LiquidScintillator 
             AlphaSLOWTIMECONSTANT          [2 ] GdDopedLS,LiquidScintillator 
             NeutronSLOWTIMECONSTANT        [2 ] GdDopedLS,LiquidScintillator 
             FASTTIMECONSTANT               [2 ] GdDopedLS,LiquidScintillator 
             RAYLEIGH                       [5 ] GdDopedLS,Acrylic,Teflon,LiquidScintillator,MineralOil 
             RINDEX                         [14] Air,GdDopedLS,Acrylic,Teflon,LiquidScintillator,Bialkali,Vacuum,Pyrex,MineralOil,Water,NitrogenGas,IwsWater,OwsWater,DeadWater 
             ABSLENGTH                      [20] PPE,Air,GdDopedLS,Acrylic,Teflon,LiquidScintillator,Bialkali,Vacuum,Pyrex,UnstStainlessSteel,StainlessSteel,
                                                 ESR,MineralOil,Water,NitrogenGas,IwsWater,ADTableStainlessSteel,Tyvek,OwsWater,DeadWater 

        Observations:
 
        #. no RAYLEIGH for water 

        """
        keymap = {
                          "RINDEX":'refractive_index',
                       "ABSLENGTH":'absorption_length',
                        "RAYLEIGH":'scattering_length',
                  "REEMISSIONPROB":'reemission_prob',
               }  


        keymat = {}
        for dmaterial in self.dae.materials:
            material = Material(dmaterial.id)   
            if dmaterial.extra is not None:
                for dkey,dval in dmaterial.extra.properties.items():

                    # record of materials that have each key 
                    if dkey not in keymat:
                        keymat[dkey] = []
                    keymat[dkey].append(material.name)

                    if dkey in keymap:
                        key = keymap[dkey]
                        material.set(key, dval[:,1], wavelengths=dval[:,0])
                        log.info("for material %s set Chroma prop %s from G4DAE prop %s vals %s " % ( material.name, key, dkey, len(dval))) 
                    else:
                        log.info("for material %s skipping G4DAE prop %s vals %s " % ( material.name, dkey, len(dval)))  
            pass 
            self.materials[material.name] = material
        pass
        log.info("G4DAE keys encountered : %s " % len(keymat))
        for dkey in sorted(keymat,key=lambda _:len(keymat[_])): 
            mats = keymat[dkey]
            print " %-30s [%-2s] %s " % ( dkey, len(mats), ",".join(map(matshorten,mats)) )


 
    def visit(self, node, debug=False):
        self.vcount += 1
        bps = list(node.boundgeom.primitives())
        bpl = bps[0]
        assert len(bps) == 1 and bpl.__class__.__name__ == 'BoundPolylist'
        tris = bpl.triangleset()

        #print node.id
        #
        #if node.id in DAENode.extra.volmap:
        #    print "node.id %s in volmap" % node.id
        #    print DAENode.extra.volmap[node.id]


        # collada meets chroma
        vertices = tris._vertex
        triangles = tris._vertex_index
        mesh = Mesh( vertices, triangles, remove_duplicate_vertices=False ) 

        # outer/inner ? triangle orientation ?
        material1 = None  
        material2 = None

        # G4DAE persists the below surface elements which 
        # both reference "opticalsurface" containing the keyed properties
        #
        # * "skinsurface" (single volumeref) 
        # * "boundarysurface" (volumeref pair) 
        #    Are the pairs always parent/child nodes ? Attempt to find them here
        #
        #
        surface = None    

        color = 0x33ffffff 
        solid = Solid( mesh, material1, material2, surface, color )

        if debug and self.vcount % 1000 == 0:
            print node.id
            print self.vcount, bpl, tris, tris.material
            print mesh
            #print mesh.assemble()
            bounds =  mesh.get_bounds()
            extent = bounds[1] - bounds[0]
            print extent





if __name__ == '__main__':
   if len(sys.argv) > 1:
       path = sys.argv[1]
   else:     
       path = '$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae'
   pass    

   DAENode.parse(path)

   #DAENode.extra.dump_skinsurface()
   #DAENode.extra.dump_skinmap()

   #DAENode.extra.dump_bordersurface()
   #DAENode.extra.dump_bordermap()

   DAENode.dump_extra_material()
  
   cc = ColladaToChroma(DAENode.orig)
   cc.convert_materials() 

   #DAENode.vwalk(cc.visit)











