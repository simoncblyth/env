#!/usr/bin/env python
"""

NB the collada to chroma functionality is kept separate 
from DAENode as that needs to operate on machines
where Chroma cannot be installed

::

   ipython collada_to_chroma.py demo.dae -i 

"""
import os, sys, logging, re
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)   # chroma has weird logging, forcing this placement 

from env.geant4.geometry.collada.daenode import DAENode
from chroma.geometry import Mesh, Solid, Material, Surface, Geometry
from chroma.loader import load_bvh



matptn = re.compile("^__dd__Materials__(\S*)0x\S{7}$")
def matshorten(name):
    m = matptn.match(name)
    if m:
        return m.group(1) 
    return name
         

class OpticalSurfaceFinish(object):
    """
    `LCG/geant4.9.2.p01/source/materials/include/G4OpticalSurface.hh`::

         61 enum G4OpticalSurfaceFinish
         62 {
         63    polished,                    // smooth perfectly polished surface
         64    polishedfrontpainted,        // smooth top-layer (front) paint
         65    polishedbackpainted,         // same is 'polished' but with a back-paint
         66    ground,                      // rough surface
         67    groundfrontpainted,          // rough top-layer (front) paint
         68    groundbackpainted            // same as 'ground' but with a back-paint
         69 };
         70 

    """
    polished = 0
    ground = 3
       

class OpticalSurfaceModel(object):
    """
    `LCG/geant4.9.2.p01/source/materials/include/G4OpticalSurface.hh`::

         71 enum G4OpticalSurfaceModel
         72 {
         73    glisur,                      // original GEANT3 model
         74    unified                      // UNIFIED model
         75 };

    """
    glisur = 0
    unified = 1


class SurfaceType(object):
    """
    `LCG/geant4.9.2.p01/source/materials/include/G4SurfaceProperty.hh`::

         66 enum G4SurfaceType
         67 {
         68    dielectric_metal,            // dielectric-metal interface
         69    dielectric_dielectric,       // dielectric-dielectric interface
         70    firsov,                      // for Firsov Process
         71    x_ray                        // for x-ray mirror process
         72 };

    """
    dielectric_metal = 0
    dielectric_dielectric = 1



class ColladaToChroma(object):
    def __init__(self, nodecls, bvh=False):
        """
        :param nodecls: typically DAENode
        """ 
        self.nodecls = nodecls
        self.bvh = bvh
        self.chroma_geometry = Geometry(detector_material=None)    # bialkali ?
        pass
        self.vcount = 0
        self.surfaces = {}
        self.materials = {}

    def convert_opticalsurfaces(self, debug=False):
        """
        Chroma surface 

        * name
        * model ? defaults to 0

        `chroma/geometry_types.h`::

           enum { SURFACE_DEFAULT, SURFACE_COMPLEX, SURFACE_WLS };

        Potentially wavelength dependent props all default to zero:

        * detect
        * absorb
        * reemit
        * reflect_diffuse
        * reflect_specular
        * eta
        * k
        * reemission_cdf

        G4DAE Optical Surface properties

        * REFLECTIVITY (the only property to be widely defined)
        * RINDEX (seems odd for a surface, looks to always be zero) 
        * SPECULARLOBECONSTANT  (set to 0.85 for a few surface)
        * BACKSCATTERCONSTANT,SPECULARSPIKECONSTANT (often present, always zero)

        """
        for dsurf in self.nodecls.extra.opticalsurface:
            if debug:
                print "%-75s %s " % (dsurf.name, dsurf )
            surface = Surface(dsurf.name)
            assert 'REFLECTIVITY' in dsurf.properties
            REFLECTIVITY = dsurf.properties['REFLECTIVITY'] 

            # guess at how to translate the Geant4 description into Chroma  
            finish = int(dsurf.finish)
            if finish == OpticalSurfaceFinish.polished:
                key = 'reflect_specular'
            elif finish == OpticalSurfaceFinish.ground:
                key = 'reflect_diffuse'
            else:
                key = None 
            pass
            assert key is not None
            log.debug("setting prop %s for surface %s " % (key, surface.name))
            surface.set(key, REFLECTIVITY[:,1], wavelengths=REFLECTIVITY[:,0])
            pass
            self.surfaces[surface.name] = surface
        pass 
        assert len(self.surfaces) == len(self.nodecls.extra.opticalsurface), "opticalsurface with duplicate names ? "
        log.debug("convert_opticalsurfaces creates %s from %s  " % (len(self.surfaces),len(self.nodecls.extra.opticalsurface))  )


    def convert_materials(self, debug=False):
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
        for dmaterial in self.nodecls.orig.materials:
            material = Material(dmaterial.id)   

            # vacuum like defaults ? is that appropriate ? what is the G4 equivalent ?
            material.set('refractive_index', 1.0)  
            material.set('absorption_length',1e6)
            material.set('scattering_length',1e6)
        
            if dmaterial.extra is not None:
                for dkey,dval in dmaterial.extra.properties.items():

                    # record of materials that have each key 
                    if dkey not in keymat:
                        keymat[dkey] = []
                    keymat[dkey].append(material.name)

                    if dkey in keymap:
                        key = keymap[dkey]
                        material.set(key, dval[:,1], wavelengths=dval[:,0])
                        log.debug("for material %s set Chroma prop %s from G4DAE prop %s vals %s " % ( material.name, key, dkey, len(dval))) 
                    else:
                        log.debug("for material %s skipping G4DAE prop %s vals %s " % ( material.name, dkey, len(dval)))  

            pass 
            self.materials[material.name] = material
        pass
        log.debug("convert_materials G4DAE keys encountered : %s " % len(keymat))
        if debug: 
            for dkey in sorted(keymat,key=lambda _:len(keymat[_])): 
                mats = keymat[dkey]
                print " %-30s [%-2s] %s " % ( dkey, len(mats), ",".join(map(matshorten,mats)) )
    

    def convert_geometry(self, nodes=None):
        """
        :param nodes: list of DAENode instances or None

        Converts DAENode/pycollada geometry into Chroma geometry.

        When `nodes=None` the entire DAENode tree is visited and converted, 
        otherwise just the listed nodes.
        """ 
        self.convert_materials() 
        self.convert_opticalsurfaces() 

        if nodes is None:  
            self.nodecls.vwalk(self.visit)
        else:
            for node in nodes:
                self.visit(node)
        pass

        log.info("ColladaToChroma convert_geometry flattening %s " % len(self.chroma_geometry.solids))

        self.chroma_geometry.flatten()
        if self.bvh:
            self.add_bvh()

    def add_bvh( self, bvh_name="default", auto_build_bvh=True, read_bvh_cache=True, update_bvh_cache=True, cache_dir=None, cuda_device=None):
        """
        As done by chroma.loader
        """
        log.info("ColladaToChroma adding BVH")
        self.chroma_geometry.bvh = load_bvh(self.chroma_geometry, 
                                            bvh_name=bvh_name,
                                            auto_build_bvh=auto_build_bvh,
                                            read_bvh_cache=read_bvh_cache,
                                            update_bvh_cache=update_bvh_cache,
                                            cache_dir=cache_dir,
                                            cuda_device=cuda_device)
        log.info("completed adding BVH")

    def find_outer_inner_materials(self, node ):
        """
        :param node: DAENode instance
        :return: Chroma Material instances for outer and inner materials

        #. Parent node material regarded as outside
        #. Current node material regarded as inside        

        Think about a leaf node to see the sense of that.

        Caveat, the meanings of "inner" and "outer" depend on 
        the orientation of the triangles that make up the surface...  
        So just adopt a convention and try to verify it later.
        """
        assert node.__class__.__name__ == 'DAENode'
        this_material = self.materials[node.matid]
        if node.parent is None:
            parent_material = this_material
            log.warning("setting parent_material to %s as parent is None for node %s " % (parent_material.name, node.id) ) 
        else:
            parent_material = self.materials[node.parent.matid]

        log.debug("find_outer_inner_materials node %s %s %s" % (node, this_material, parent_material))
        return parent_material, this_material
        
    def find_skinsurface(self, node):
        """
        :param node: DAENode instance
        :return: G4DAE Surface instance corresponding to G4LogicalSkinSurface if one is available for the LV of the current node
        """
        assert node.__class__.__name__ == 'DAENode'
        lvid = node.lv.id
        skin = self.nodecls.extra.skinmap.get(lvid, None)
        if skin is not None:
            assert len(skin) == 1, "ambigous skin for lvid %s " % lvid 
            skin = skin[0]

        return skin
           
    def find_bordersurface(self, node):
        """
        :param node: DAENode instance
        :return: G4DAE Surface instance corresponding to G4LogicalBorderSurface 
                 if one is available for the PVs of the current node and its parent

        Ambiguity bug makes this difficult
        """
        assert node.__class__.__name__ == 'DAENode'
        pass
        #pvid = node.pv.id
        #ppvid = node.parent.pv.id
        #border = self.nodecls.extra.bordermap.get(pvid, None)
        return None


    def find_surface(self, node):
        """
        :param node: DAENode instance
        :return Chroma Surface instance or None:

        G4DAE persists the below surface elements which 
        both reference "opticalsurface" containing the keyed properties
        
        * "skinsurface" (single volumeref, ref by lv.id)
        * "boundarysurface" (physvolref ordered pair, identified by pv1.id,pv2.id) 
          
        The boundary pairs are always parent/child nodes in dyb Near geometry, 
        they could in principal be siblings.
        """
        assert node.__class__.__name__ == 'DAENode'
        skin = self.find_skinsurface( node )
        border = self.find_bordersurface( node )
        dsurf = filter(None,[skin, border])
        assert len(dsurf)<2, "Not expecting both skin %s and border %s surface for the same node %s "  % (skin, border, node)
        if len(dsurf) == 1:
            dsurface = dsurf[0]
            log.debug("found dsurface %s for node %s " % (dsurface, node ))
            surface = self.surfaces.get(dsurface.name, None)
            assert surface is not None, "found dsurface %s without corresponding chroma surface for node %s " % ( dsurface, node.id) 
        else:
            surface = None  
        pass 
        return surface


    def visit(self, node, debug=False):
        """
        :param node: DAENode instance

        DAENode instances and their pycollada underpinnings meet chroma here
        """
        assert node.__class__.__name__ == 'DAENode'
        self.vcount += 1
        if self.vcount < 10:
            log.debug("visit : vcount %s node.index %s node.id %s " % ( self.vcount, node.index, node.id ))

        bps = list(node.boundgeom.primitives())
        bpl = bps[0]
        assert len(bps) == 1 and bpl.__class__.__name__ == 'BoundPolylist'
        tris = bpl.triangleset()

        vertices = tris._vertex

        triangles = tris._vertex_index

        mesh = Mesh( vertices, triangles, remove_duplicate_vertices=False ) 

        material2, material1 = self.find_outer_inner_materials(node)   

        surface = self.find_surface( node )

        color = 0x33ffffff 

        solid = Solid( mesh, material1, material2, surface, color )
        self.chroma_geometry.add_solid( solid )


        if debug and self.vcount % 1000 == 0:
            print node.id
            print self.vcount, bpl, tris, tris.material
            print mesh
            #print mesh.assemble()
            bounds =  mesh.get_bounds()
            extent = bounds[1] - bounds[0]
            print extent



def daeload(path=None, bvh=False ):
   """
   :param path:
   :return Chroma Geometry instance:

   This is invoked by chroma.loader.load_geometry_from_string when
   the string ends with ".dae".  This allows the standard chroma-cam 
   to be used with COLLADA geometries.

   TODO: add nodespec capabilities here to allow loading partial geometries from chroma-cam 
   """
   if path is None:
       path = os.environ['DAE_NAME']

   log.info("daeload path %s " % (path))

   if len(DAENode.registry) > 0:
       log.debug("skipping parse as already have %s nodes " % len(DAENode.registry))
   else:
       DAENode.parse(path)
   pass

   cc = ColladaToChroma(DAENode, bvh=bvh )  
   cc.convert_geometry()

   return cc.chroma_geometry



    




if __name__ == '__main__':
   if len(sys.argv) > 1:
       path = sys.argv[1]
   else:     
       path = '$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae'
   pass    

   daeload(path)



