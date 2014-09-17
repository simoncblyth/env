#!/usr/bin/env python
"""
TODO
=====

#. split DAENode up into creation and querying portions


Geant4 level interpretation of G4DAEWrite exported pycollada geometry
=======================================================================

Attempt to create the DAENode heirarcy out of a raw collada traverse 
without monkeying around.


Debug Usage
-------------

::

    daenode.sh --ipy --surface



Usage Examples
----------------

::

   daenode.py --tree 0 > 0.txt            # single line just the world volume, as no recursion by default
   daenode.py --tree 0___2 > 0___2.txt    # 


Use from ipython
-----------------

::

    In [64]: import logging
    In [65]: logging.basicConfig(level=logging.INFO)        ## logging is needed to see any output
    In [60]: from env.geant4.geometry.collada.daenode import DAENode, Defaults
    In [61]: Defaults.daepath
    Out[61]: '$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae'
    In [62]: DAENode.parse(Defaults.daepath)
    In [62]: DAENode.parse("g4_00.dae")
    In [66]: DAENode.summary()
    INFO:env.graphics.collada.pycollada.daenode:registry 12230 
    INFO:env.graphics.collada.pycollada.daenode:lookup 12230 
    INFO:env.graphics.collada.pycollada.daenode:idlookup 12230 
    INFO:env.graphics.collada.pycollada.daenode:ids 12230 
    INFO:env.graphics.collada.pycollada.daenode:rawcount 36690 
    INFO:env.graphics.collada.pycollada.daenode:created 12230 
    INFO:env.graphics.collada.pycollada.daenode:root   top.0             

    In [71]: node = DAENode.get("1000")
    INFO:env.graphics.collada.pycollada.daenode:arg 1000 => indices [1000] => node   __dd__Geometry__RPC__lvRPCGasgap23--pvStrip23Array--pvStrip23ArrayOne..6--pvStrip23Unit0xa8c15c8.46             __dd__Materials__MixGas0x8837740  

    In [72]: node
    Out[72]:   __dd__Geometry__RPC__lvRPCGasgap23--pvStrip23Array--pvStrip23ArrayOne..6--pvStrip23Unit0xa8c15c8.46             __dd__Materials__MixGas0x8837740 

    In [73]: node.children
    Out[73]: []

    In [74]: node.ancestors
    Out[74]: <bound method DAENode.ancestors of   __dd__Geometry__RPC__lvRPCGasgap23--pvStrip23Array--pvStrip23ArrayOne..6--pvStrip23Unit0xa8c15c8.46             __dd__Materials__MixGas0x8837740 >

    In [75]: node.ancestors()
    Out[75]: 
    [  __dd__Geometry__RPC__lvRPCBarCham23--pvRPCGasgap230xa8c1918.46             __dd__Materials__Air0x8838278 ,
       __dd__Geometry__RPC__lvRPCFoam--pvBarCham23Array--pvBarCham23ArrayOne..1--pvBarCham23Unit0xa8c1b80.23             __dd__Materials__Bakelite0x8838888 ,
       __dd__Geometry__RPC__lvRPCMod--pvRPCFoam0xa8c1d58.23             __dd__Materials__Foam0x8838a98 ,
       __dd__Geometry__RPC__lvNearRPCRoof--pvNearUnSlopModArray--pvNearUnSlopModOne..4--pvNearUnSlopMod..4--pvNearSlopModUnit0xa8c3868.0             __dd__Materials__Aluminium0x88391b8 ,
       __dd__Geometry__Sites__lvNearHallTop--pvNearRPCRoof0xa8d3ca8.0             __dd__Materials__Air0x8838278 ,
       __dd__Geometry__Sites__lvNearSiteRock--pvNearHallTop0xaa8ace0.0             __dd__Materials__Air0x8838278 ,
       __dd__Structure__Sites__db-rock0xaa8b0f8.0             __dd__Materials__Rock0x8868188 ,
       top.0             - ]



Heavy Geometries timeout through web interface
-------------------------------------------------

::

   daenode.py -e -s 1___100    > 1___100.dae 
   daenode.py -e -s 3148___100 > 3148___100.dae    
   daenode.py -e -s 3148___6   > 3148___6.dae 

Notable Volumes
----------------


* http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xaa88da8.0.html

  * needs sibling culling

* http://belle7.nuu.edu.tw/dae/tree/__dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xa9883f0.0.html
 
  * 2 ADs split 


Node Dumping
------------------

Access the geometry via a web server (using webpy) from curl or browser
avoiding the overhead of parsing/traversing the entire .dae just 
to look at a selection of volumes::

     http://localhost:8080/node/1000:1100
     http://localhost:8080/node/0:10,100:110?ancestors=1

Tree Dumping
--------------

Text presentation of volume tree::

     http://localhost:8080/tree/6370
     http://localhost:8080/tree/3154
     http://localhost:8080/tree/__dd__Geometry__AD__lvADE--pvSST0xa906040.0
     http://localhost:8080/tree/__dd__Geometry__AD__lvADE--pvSST0xa906040.1

     http://localhost:8080/tree/2431___5?ancestors=1     # the ___5 specifies maxdepth of 5 from PV 2431 


Equivalent to commandline::

     ./daenode.py -t __dd__Geometry__AD__lvADE--pvSST0xa906040.0


From CLI remember to escape the ampersand::

    curl http://localhost:8080/node/1000?ancestors=1\&other=yes
    curl http://localhost:8080/node/__dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..2--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xb35ffb0.1?ancestors=1
    curl http://localhost:8080/node/__dd__Geometry__AD__lvSST--pvOIL0xb36eb48.1?ancestors=1

Subcopy
---------

::

    ./daenode.py -e -s __dd__Geometry__AD__lvOAV--pvLSO0xa8d68e0.0
    ./daenode.py -e -s __dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8d92d8.0
    ./daenode.py -e -s top.0


Alternatively to avoid the overhead of repeating the initial parse use the `--webserver` option and subcopy volumes
selected by uniqued pvname (with the .0 .1 etc..) thru commandline or browser::

    curl -O http://localhost:8080/geom/__dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8d92d8.0


Depth Restricted
~~~~~~~~~~~~~~~~~~

The maxdepth controls how much recursion from the target volume is included in the subcopy. 
The default is the entire tree beneath the target volume.
A maxdepth of zero inhibits any recursion, so just the single targetted volume is copied. 

::

    daenode.py --maxdepth 0 -s 0
    curl http://localhost:8080/geom/3199.dae?maxdepth=0
    curl -sO http://localhost:8080/geom/3199___0.dae        
 
daenode.py eats its own dogfood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:collada blyth$ daenode.py --daepath=0000000.xml --geom --subpath=0000000_dogfood.xml --daesave 0
    2013-10-29 12:17:26,499 env.graphics.collada.pycollada.daenode INFO     /Users/blyth/env/bin/daenode.py
    ...
    2013-10-29 12:18:33,953 env.graphics.collada.pycollada.daenode INFO     daesave to 0000000_dogfood.xml 
    simon:collada blyth$ diff 0000000.xml 0000000_dogfood.xml
    3,4c3,4
    <     <created>2013-10-28T17:20:00.303554</created>
    <     <modified>2013-10-28T17:20:00.303589</modified>
    ---
    >     <created>2013-10-29T12:17:40.367896</created>
    >     <modified>2013-10-29T12:17:40.367933</modified>



Webserver flakiness
---------------------

From webserver, there is a tendency to loose the registry after any type of error is encountered ? 
Forcing a restart of the webserver.  Hmm perhaps webpy can adopt a forking server approach to avoid 
this issue and hence improve robustness.

Possibly webpy has some session timeout, as seems to happen when make a query after some delay.


cProfile running
-----------------

::

    cd $LOCAL_BASE/env/graphics/collada 
    simon:collada blyth$ python -m cProfile -o daenode.cprofile $(which daenode.py) --daesave --subcopy -O 000.xml 0 
    2013-10-28 14:06:38,504 env.graphics.collada.pycollada.daenode INFO     /Users/blyth/env/bin/daenode.py
    2013-10-28 14:06:38,509 env.graphics.collada.pycollada.daenode INFO     DAENode.parse pycollada parse /usr/local/env/geant4/geometry/xdae/g4_01.dae 

    simon:collada blyth$ gprof2dot.py -f pstats daenode.cprofile | dot -Tsvg -o daenode.svg

Profiling points to 35% from multiarray.fromstring, especially in geometry load (50%). 
Look into deepcopying rather than going back to XML. 


Visualizing Collada Files
--------------------------

Best way so far is with "meshtool", a python based viewer from the pycollada author
which is built upon Panda3D/OpenGL::

     meshtool-view http://localhost:8080/subcopy/3___3.dae 

I patched this to allow grabbing geometry from a url, allowing viewing geometry 
dynamically pulled off a webserver.

::

    simon:collada blyth$ meshtool-
    simon:collada blyth$ t meshtool-view
    meshtool-view is a function
    meshtool-view () 
    { 
        meshtool --load_collada $* --viewer
        }
    simon:collada blyth$ t meshtool
    meshtool is a function
    meshtool () 
    { 
        /usr/bin/python -c "from meshtool.__main__ import main ; main() " $*
    }


Other ways, include:

#. pycollada-view   # based on daeview, which is finnicky : possibly due to fixed light/view/camera positioning 
#. blender          # slow for large geometries, awful interface 
#. Preview.app/Xcode.app on newer Macs


finding promising volumes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    collada-cd ; vi vnodetree.txt   # 
    curl -sO http://localhost:8080/geom/3154___3.dae
    meshtool-view 3154___3.dae

    curl -sO http://localhost:8080/subcopy/


    http://localhost:8080/node/3154?ancestors=1&children=1


rpc 
~~~~

::

    http://localhost:8080/tree/2431___5?ancestors=1   # text presentation of node tree 
    curl -sO http://localhost:8080/geom/2431___5.dae
    meshtool-view 2431___5.dae



"""

import numpy as np
import collada

# CAUTION MONKEY PATCH DIDDLING TRANSFORMATION MATRIX : not needed for DAE exported from 20131119-1632
#from monkey_matrix_load import _monkey_matrix_load
#collada.scene.MatrixTransform.load = staticmethod(_monkey_matrix_load)   

from collada.xmlutil import etree as ET
from collada.xmlutil import writeXML, COLLADA_NS, E
from collada.common import DaeObject


tag = lambda _:str(ET.QName(COLLADA_NS,_))

# relying on a load signature that is not uniformly followed, works for Transforms, but too much action-at-a-distance
#collada.common.DaeObject.copy = lambda self:self.load(collada, self.xmlnode )  
#collada.scene.Transform.copy = lambda self:self.load(collada, self.xmlnode )  

# disable saves, which update the xmlnode, as the preexisting xmlnode for 
# the basis objects are being copied anyhow
collada.geometry.Geometry.save = lambda _:_
collada.material.Material.save = lambda _:_

tostring_ = lambda _:ET.tostring(getattr(_,'xmlnode'))

import sys, os, logging, hashlib, copy, re
log = logging.getLogger(__name__)
from StringIO import StringIO

try:
    import web 
except ImportError:
    web = None



def present_geometry( bg ):
    out = []
    out.append(bg)
    for bp in bg.primitives():
        out.append("nvtx:%s" % len(bp.vertex))
        out.append(bp.vertex)
    return out


class DAENode(object):
    registry = []
    lookup = {}
    idlookup = {}
    pvlookup = {}
    lvlookup = {}
    ids = set()
    created = 0
    root = None
    rawcount = 0
    verbosity = 1   # 0:almost no output, 1:one liners, 2:several lines, 3:extreme  
    argptn = re.compile("^(\S*)___(\d*)")   

    @classmethod
    def summary(cls):
        log.info("registry %s " % len(cls.registry) )
        log.info("lookup %s " % len(cls.lookup) )
        log.info("idlookup %s " % len(cls.idlookup) )
        log.info("ids %s " % len(cls.ids) )
        log.info("rawcount %s " % cls.rawcount )
        log.info("created %s " % cls.created )
        log.info("root %s " % cls.root )

    @classmethod
    def dump(cls):
        cls.extra.dump_skinsurface()
        cls.extra.dump_skinmap()
        cls.extra.dump_bordersurface()
        cls.extra.dump_bordermap()
        cls.dump_extra_material()

    @classmethod
    def idmap_parse( cls, path ):
        """
        Read g4_00.idmap file that maps between volume index 
        and sensdet identity, ie the PmtId
        """
        if os.path.exists(path):
            from idmap import IDMap
            idmap = IDMap(path)
            log.info("idmap exists %s entries %s " % (path, len(idmap)))
        else:
            log.warn("no idmap found at %s " % path )
            idmap = None
        pass
        return idmap 

    @classmethod
    def idmaplink(cls, idmap ):
        if idmap is None:
            log.warn("skip idmaplink ")
            return 
        pass
        log.info("linking DAENode with idmap %s identifiers " % len(idmap)) 
        assert len(cls.registry) == len(idmap), ( len(cls.registry), len(idmap))
        for index, node in enumerate(cls.registry):
            node.channel_id = idmap[index]
            #if index % 100 == 0:
            #    print index, node.channel_id, node, node.__class__


    @classmethod
    def parse( cls, path ):
        """
        :param path: to collada file

        #. `collada.Collada` parses the .dae 
        #. a list of bound geometry is obtained from `dae.scene.objects`
        #. `DAENode.recurse` traverses the raw pycollada node tree, creating 
           an easier to navigate DAENode heirarchy which has one DAENode per bound geometry  
        #. cross reference between the bound geometry list and the DAENode tree

        """
        path = os.path.expandvars(path)
        log.debug("DAENode.parse pycollada parse %s " % path )

        base, ext = os.path.splitext(path)
        idmap = cls.idmap_parse( base + ".idmap" ) # path with .idmap instead of .dae

        dae = collada.Collada(path)
        log.debug("pycollada parse completed ")
        boundgeom = list(dae.scene.objects('geometry'))
        top = dae.scene.nodes[0]
        log.debug("pycollada binding completed, found %s  " % len(boundgeom))
        log.debug("create DAENode heirarchy ")
        cls.orig = dae
        cls.recurse(top)
        #cls.summary()
        cls.indexlink( boundgeom )
        cls.idmaplink( idmap )

        cls.parse_extra_surface( dae )
        cls.parse_extra_material( dae )

    @classmethod
    def parse_extra_surface( cls, dae ):
        """
        """
        log.debug("collecting opticalsurface/boundarysurface/skinsurface info from library_nodes/extra/")
        library_nodes = dae.xmlnode.find(".//"+tag("library_nodes"))
        extra = library_nodes.find(tag("extra"))
        assert extra is not None
        cls.extra = DAEExtra.load(collada, {}, extra) 

    @classmethod
    def parse_extra_material( cls, dae ):
        log.debug("collecting extra material properties from library_materials/material/extra ")
        nextra = 0 
        for material in dae.materials:
            extra = material.xmlnode.find(tag("extra"))
            if extra is None:
                material.extra = None
            else:
                nextra += 1
                material.extra = MaterialProperties.load(collada, {}, extra)
            pass 
        log.debug("loaded %s extra elements with MaterialProperties " % nextra )             

    @classmethod
    def dump_extra_material( cls ):
        log.info("dump_extra_material")
        for material in cls.orig.materials:
            print material
            if material.extra is not None:
                print material.extra 
 

    @classmethod
    def recurse(cls, node , ancestors=[], extras=[] ):
        """
        This recursively visits 12230*3 = 36690 Nodes.  
        The below pattern of triplets of node types is followed precisely, due to 
        the node/instance_node/instance_geometry layout adopted for the dae file.

        The triplets are collected into DAENode on every 3rd leaf node.

        ::

            1 0 <MNode top transforms=0, children=1>
            2 1 <NodeNode node=World0xb50dfb8>
            3 2 <MGeometryNode geometry=WorldBox0xb342f60>

            4 2 <MNode __dd__Structure__Sites__db-rock0xb50e0f8 transforms=1, children=1>
            5 3 <NodeNode node=__dd__Geometry__Sites__lvNearSiteRock0xb50de78>
            6 4 <MGeometryNode geometry=near_rock0xb342e30>

            7 4 <MNode __dd__Geometry__Sites__lvNearSiteRock--pvNearHallTop0xb50dce0 transforms=1, children=1>
            8 5 <NodeNode node=__dd__Geometry__Sites__lvNearHallTop0xb356a70>
            9 6 <MGeometryNode geometry=near_hall_top_dwarf0x92eee48>

            10 6 <MNode __dd__Geometry__Sites__lvNearHallTop--pvNearTopCover0xb356790 transforms=1, children=1>
            11 7 <NodeNode node=__dd__Geometry__PoolDetails__lvNearTopCover0xb342fe8>
            12 8 <MGeometryNode geometry=near_top_cover_box0x92ecf48>

        """
        cls.rawcount += 1

        children = []
        xtras = []  
        if hasattr(node, 'children'):
            for c in node.children:
                if isinstance(c, collada.scene.ExtraNode):
                    xtras.append(c.xmlnode)
                else:
                    children.append(c) 
                pass
            pass     
        pass
        #log.info("node: %s " % node )
        #log.info("xtras: %s " % xtras )
        #log.info("extras: %s " % extras )

        if len(children) == 0: #leaf formation, gets full ancestry to go on
            cls.make( ancestors + [node], extras + xtras )
        else:
            for child in children:
                cls.recurse(child, ancestors = ancestors + [node] , extras = extras + xtras )

    @classmethod
    def indexget(cls, index):
        return cls.registry[index]

    @classmethod
    def indexgets(cls, indices):
        return [cls.registry[index] for index in indices]

    @classmethod
    def idget(cls, id):
        return cls.idlookup.get(id, None)

    @classmethod
    def get(cls, arg ):
        indices = cls.interpret_ids(arg)
        index = indices[0]
        node = cls.registry[index]
        log.info("arg %s => indices %s => node %s " % ( arg, indices, node ))
        return node

    @classmethod
    def getall(cls, arg, path=None):
        if not path is None:
            cls.init(path)
        pass
        indices = cls.interpret_ids(arg)
        return [cls.registry[index] for index in indices]

    @classmethod
    def interpret_arg(cls, arg):
        """
        Interpret arguments like:

        #. 0
        #. __dd__some__path.0
        #. __dd__some__path.0___0
        #. 0___0
        #. top.0___0

        Where the triple underscore ___\d* signified the maxdepth to recurse.
        """
        match = cls.argptn.match(arg)
        if match:
            arg, maxdepth = match.groups()
        else:
            maxdepth = 0  # default to zero
        return arg, maxdepth

    @classmethod
    def interpret_ids(cls, arg_, dedupe=True):
        """
        Interprets an "," delimited string like 0:10,400:410,300,40,top.0
        into a list of integer DAENode indices. Where each element
        can use one of the below forms. 

        Listwise::

              3153:3160
              3153:        # means til the end of the registry 

        Shortform treewise, only allows setting mindepth to 0,1::

              3153-     # mindepth 0
              3153+     # mindepth 1 

        Longform treewise, allows mindepth/maxdepth spec::

              3153_1.5  # mindepth 1, maxdepth 5

        Intwise::

              0

        Identifier::

              top.0

        """
        indices = []
        for arg in arg_.split(","):
            prelast, last = arg[:-1], arg[-1]

            listwise = ":" in arg
            treewise_short = last in ("-","+") 
            treewise_long  = arg.count("_") == 1
            intwise = arg.isdigit()

            assert len(filter(None,[listwise,treewise_short,treewise_long,intwise]))<=1, "mixing forms not allowed" 

            if treewise_short or treewise_long:
                pass
                if treewise_short:
                    pass
                    baseindex= prelast
                    mindepth = 0 if last == "-" else 1
                    maxdepth = 100
                    pass
                elif treewise_long:
                    pass
                    elem = arg.split("_")
                    assert len(elem) == 2
                    baseindex = elem[0]
                    mindepth, maxdepth = map(int,elem[1].split(".")) 
                else:
                    assert 0
                treewise_indices = cls.progeny_indices(baseindex, mindepth=mindepth, maxdepth=maxdepth )
                indices.extend(treewise_indices) 

            elif listwise:
                pass
                if last == ":":
                    arg = "%s:%s" % (prelast, len(cls.registry)-1) 
                pass
                listwise_indices=range(*map(int,arg.split(":")))
                indices.extend(listwise_indices)
                pass

            elif intwise:

                indices.append(int(arg))

            else:
                # node lookup by identifier like top.0
                if "___" in arg:
                    arg = arg.split("___")[0]   # get rid of the maxdepth indicator eg "___0"
                node = cls.idlookup.get(arg,None)
                if node:
                    indices.append(node.index)
                else:
                    log.warn("failed to lookup DAENode for arg %s " % arg)
                pass
            pass
        return list(set(indices)) if dedupe else indices


    @classmethod
    def init(cls, path ):
        if path is None:
            path = os.environ['DAE_NAME']
         
        if len(cls.registry) == 0:
            cls.parse(path)


    @classmethod
    def find_uid(cls, bid, decodeNCName=False):
        """
        :param bid: basis ID
        :param decodeNCName: more convenient not to decode for easy URL/cmdline  arg passing without escaping 

        Find a unique id for the emerging DAENode
        """
        if decodeNCName:
            bid = bid.replace("__","/").replace("--","#").replace("..",":")
        uid = None
        count = 0 
        while uid is None or uid in cls.ids: 
            uid = "%s.%s" % (bid,count)
            count += 1
        pass
        cls.ids.add(uid)
        return uid 

    @classmethod
    def make(cls, nodepath, extras ):
        """
        Creates `DAENode` instances and positions them within the volume tree
        by setting the `parent` and `children` attributes.

        A digest keyed lookup gives fast access to node parents,
        the digest represents a path through the tree of nodes.
        """
        node = cls(nodepath, extras )
        if node.index == 0:
            cls.root = node

        cls.registry.append(node)
        cls.idlookup[node.id] = node   

        # a list of nodes for each pv.id, need for a list is not so obvious, maybe GDML PV identity bug ?
        pvid = node.pv.id
        if pvid not in cls.pvlookup:
            cls.pvlookup[pvid] = []
        cls.pvlookup[pvid].append(node) 

        # list of nodes for each lv.id, need for a list is obvious
        lvid = node.lv.id
        if lvid not in cls.lvlookup:
            cls.lvlookup[lvid] = []
        cls.lvlookup[lvid].append(node) 

   
        cls.lookup[node.digest] = node   
        cls.created += 1

        parent = cls.lookup.get(node.parent_digest)
        node.parent = parent
        if parent is None:
            if node.id == "top.0":
                pass
            elif node.id == "top.1":
                log.info("root node name %s indicates have parsed twice " % node.id )
            else:
                log.fatal("failed to find parent for %s (failure expected only for root node)" % node )
                assert 0
        else:
            parent.children.append(node)  

        if cls.created % 1000 == 0:
            log.debug("make %s : [%s] %s " % ( cls.created, id(node), node ))
        return node

    @classmethod
    def pvfind(cls, pvid ):
        return cls.pvlookup.get(pvid,[])

    @classmethod
    def lvfind(cls, lvid ):
        return cls.lvlookup.get(lvid,[])

    @classmethod
    def walk(cls, node=None, depth=0):
        if node is None:
            cls.wcount = 0
            node=cls.root

        cls.wcount += 1 
        if cls.wcount % 100 == 0:
            log.info("walk %s %s %s " % ( cls.wcount, vdepth, node ))
            if hasattr(node,'boundgeom'):
                print node.boundgeom

        for subnode in node.children:
            cls.walk(subnode, depth+1)


    @classmethod
    def vwalk(cls, visit_=lambda node:None, node=None, depth=0):
        if node is None:
            node=cls.root
        visit_(node) 
        for subnode in node.children:
            cls.vwalk(visit_=visit_, node=subnode, depth=depth+1)

    @classmethod
    def dwalk(cls, visit_=lambda node:None, node=None, depth=0):
        if node is None:
            node=cls.root
        visit_(node, depth) 
        for subnode in node.children:
            cls.dwalk(visit_=visit_, node=subnode, depth=depth+1)


    @classmethod
    def vwalks(cls, visits=[], node=None, depth=0):
        if node is None:
            node=cls.root
        for visit_ in visits:
            visit_(node) 
        for subnode in node.children:
            cls.vwalks(visits=visits, node=subnode, depth=depth+1)

    @classmethod
    def progeny_nodes(cls, baseindex=None, mindepth=0, maxdepth=100):
        """
        :param baseindex: of base node of interest within the tree
        :param mindepth:  0 includes basenode, 1 will start from children of basenode
        :param maxdepth:  0 includes basenode, 1 will start from children of basenode
        :return: all nodes in the tree below and including the base node
        """ 
        nodes = []
        basenode = cls.root if baseindex is None else cls.get(str(baseindex))
        pass
        def visit_(node, depth):
            if mindepth <= depth <= maxdepth:
                nodes.append(node)
            pass
        pass
        cls.dwalk(visit_=visit_, node=basenode )  # recursive walk 
        return nodes

    @classmethod
    def progeny_indices(cls, baseindex=None, mindepth=0, maxdepth=100):
        """
        :param baseindex: of base node of interest within the tree
        :param mindepth:  0 includes basenode, 1 will start from children of basenode
        :param maxdepth:  0 includes basenode, 1 will start from children of basenode
        :return: all nodes in the tree below and including the base node
        """ 
        indices = []
        basenode = cls.root if baseindex is None else cls.get(str(baseindex))
        pass
        def visit_(node, depth):
            if mindepth <= depth <= maxdepth:
                indices.append(node.index)
            pass
        pass
        cls.dwalk(visit_=visit_, node=basenode )  # recursive walk 
        return indices




    @classmethod
    def md5digest(cls, nodepath ):
        """
        Use of id means that will change from run to run. 
        """
        dig = ",".join(map(lambda _:str(id(_)),nodepath))
        dig = hashlib.md5(dig).hexdigest() 
        return dig


    @classmethod
    def indexlink(cls, boundgeom ):
        """
        index linked cross referencing

        For this to be correct the ordering that pycollada comes
        up with for the boundgeom must match the DAENode ordering

        The geometry id comparison performed is a necessary condition, 
        but it does not imply correctness of the cross referencing due
        to a lot of id recycling.
        """
        log.info("index linking DAENode with boundgeom %s volumes " % len(boundgeom)) 
        assert len(cls.registry) == len(boundgeom), ( len(cls.registry), len(boundgeom))
        for vn,bg in zip(cls.registry,boundgeom):
            vn.boundgeom = bg
            vn.matdict = vn.get_matdict()
            bg.daenode = vn
            assert vn.geo.geometry.id == bg.original.id   
        log.debug("index linking completed")    


    @classmethod
    def _metadata(cls, extras):
        """
        :param extras: list of xmlnode of extra elements

        Interpret extra/meta/* text elements, converting into a dict 
        """
        d = {}
        extra = None
        if len(extras)>0:
            extra = extras[-1]
        if not extra is None:
            meta = extra.find("{%s}meta" % COLLADA_NS ) 
            for elem in meta.findall("*"):
                tag = elem.tag[len(COLLADA_NS)+2:]
                d[tag] = elem.text 
        return d



    def ancestors(self,andself=False):
        """
        ::

            In [35]: print VNode.registry[1000]
            VNode(17,19)[1000,__dd__Geometry__RPC__lvRPCGasgap23--pvStrip23Array--pvStrip23ArrayOne..6--pvStrip23Unit0xb3445c8.46]

            In [36]: for _ in VNode.registry[1000].ancestors():print _
            VNode(15,17)[994,__dd__Geometry__RPC__lvRPCBarCham23--pvRPCGasgap230xb344918.46]
            VNode(13,15)[993,__dd__Geometry__RPC__lvRPCFoam--pvBarCham23Array--pvBarCham23ArrayOne..1--pvBarCham23Unit0xb344b80.23]
            VNode(11,13)[972,__dd__Geometry__RPC__lvRPCMod--pvRPCFoam0xb344d58.23]
            VNode(9,11)[971,__dd__Geometry__RPC__lvNearRPCRoof--pvNearUnSlopModArray--pvNearUnSlopModOne..4--pvNearUnSlopMod..4--pvNearSlopModUnit0xb346868.0]
            VNode(7,9)[88,__dd__Geometry__Sites__lvNearHallTop--pvNearRPCRoof0xb356ca8.0]
            VNode(5,7)[2,__dd__Geometry__Sites__lvNearSiteRock--pvNearHallTop0xb50dce0.0]
            VNode(3,5)[1,__dd__Structure__Sites__db-rock0xb50e0f8.0]
            VNode(1,3)[0,top.0]

        """
        anc = []
        if andself:
            anc.append(self)
        p = self.parent
        while p is not None:
            anc.append(p)
            p = p.parent
        return anc    

    @classmethod
    def format_keys(cls, fmt):
        ptn = re.compile("\%\((\S*)\)s") 
        return ptn.findall(fmt)

    def format(self, fmt, keys=None ):
        if keys is None: 
            keys = self.format_keys(fmt)
        pass     
        nom = self.metadata
        bgm = self.boundgeom_metadata()
        pass
        d = {}
        for key in keys:
            x = key[0:2]
            k = key[2:]
            if x == 'p_':
                d["p_%s" % k] = getattr(self, k, "-")
            elif x == 'n_':
                d["n_%s" % k] = nom.get(k,"-")
            elif x == 'g_':
                d["g_%s" % k] = bgm.get(k,"-")
            else:
                pass
            pass    
        return fmt % d
 
    def __init__(self, nodepath, extras):
        """
        :param nodepath: list of node instances identifying all ancestors and the leaf geometry node
        :param rootdepth: depth 
        :param leafdepth: 

        Currently `rootdepth == leafdepth - 2`,  making each DAENode be constructed out 
        of three raw recursion levels.

        `digest` represents the identity of the specific instances(memory addresses) 
        of the nodes listed in the nodepath allowing rapid ancestor comparison
        """
        assert len(nodepath) >= 3
        leafdepth = len(nodepath)
        rootdepth = len(nodepath) - 2

        pv, lv, geo = nodepath[-3:]
        assert pv.__class__.__name__ in ('Node'), (pv, nodepath)
        assert lv.__class__.__name__ in ('NodeNode'), (lv, nodepath)
        assert geo.__class__.__name__ in ('GeometryNode'), (geo, nodepath)

        self.children = []
        self.metadata = self._metadata(extras)
        self.leafdepth = leafdepth
        self.rootdepth = rootdepth 
        self.digest = self.md5digest( nodepath[0:leafdepth-1] )
        self.parent_digest = self.md5digest( nodepath[0:rootdepth-1] )
        self.matdict = {}  # maybe filled in during index linking 

        # formerly stored ids rather than instances to allow pickling 
        self.pv = pv
        self.lv = lv   
        self.geo = geo
        #self.geo = geo.geometry.id
        pass
        self.id = self.find_uid( pv.id , False)
        self.index = len(self.registry)


    def get_matdict(self):
        assert hasattr(self, 'boundgeom'), "matdict requires associated boundgeom "
        msi = self.boundgeom.materialnodebysymbol.items()
        assert len(msi) == 1 
        symbol, matnode= msi[0]
        return dict(matid=matnode.target.id, symbol=symbol, matnode=matnode)
 
    matnode  = property(lambda self:self.matdict.get('matnode',None))
    matid  = property(lambda self:self.matdict.get('matid',None))
    symbol = property(lambda self:self.matdict.get('symbol',None))

    def primitives(self):
        if not hasattr(self, 'boundgeom'):
            return []
        bg = self.boundgeom
        lprim = list(bg.primitives())
        ret = ["nprim %s " % len(lprim)]
        for bp in lprim:
            ret.append("bp %s nvtx %s " % (str(bp),len(bp.vertex)))
            ret.append("vtxmax %s " % str(bp.vertex.max(axis=0)))
            ret.append("vtxmin %s " % str(bp.vertex.min(axis=0)))
            ret.append("vtxdif %s " % str(bp.vertex.max(axis=0)-bp.vertex.min(axis=0)))
        return ret

    def boundgeom_metadata(self):
        if not hasattr(self, 'boundgeom'):
            return {}
        extras = self.boundgeom.original.xmlnode.findall(".//{%s}extra" % COLLADA_NS )
        return self._metadata(extras)

    def __str__(self):
        lines = []
        if self.verbosity > 0:
            lines.append("  %s             %s " % (self.id, self.matdict.get('matid',"-") ) )
        if self.verbosity > 1:    
            lines.append("DAENode(%s,%s)[%s]    %s             %s " % (self.rootdepth,self.leafdepth,self.index, self.id, self.matdict.get('matid',"-") ) )
            lines.append("    pvid         %s " % self.pv.id )
            lines.append("    lvid         %s " % self.lv.id )
            lines.append("    ggid         %s " % self.geo.geometry.id )
        if self.verbosity > 2:    
            lines.extend(self.primitives())
        return "\n".join(lines)

    __repr__ = __str__



# follow the pycollada pattern for extra nodes


hc_over_GeV = 1.2398424468024265e-06 # h_Planck * c_light / GeV / nanometer #  (approx, hc = 1240 eV.nm )  
hc_over_MeV = hc_over_GeV*1000.
hc_over_eV  = hc_over_GeV*1.e9


def as_optical_property_vector( s, xunit='MeV', yunit=None ):
    """ 
    Units of the input string first column as assumed to be MeV, 
    (G4MaterialPropertyVector raw numbers photon energies are in units of MeV)
    these are converted to nm and the order is reversed in the returned
    numpy array.
        
    :param s: string with space delimited floats representing a G4MaterialPropertyVector 
    :return: numpy array with nm
    """ 
    # from chroma/demo/optics.py 
    a = np.fromstring(s, dtype=float, sep=' ')
    assert len(a) % 2 == 0
    b = a.reshape((-1,2))[::-1]   ## reverse energy, for ascending wavelength nm

    if yunit is None or yunit in ('','mm'):
        val = b[:,1]
    elif yunit == 'cm':
        val = b[:,1]*10.
    else:   
        assert 0, "unexpected yunit %s " % yunit
        
    energy = b[:,0]
    
    hc_over_x = 0
    if xunit=='MeV':
        hc_over_x  = hc_over_MeV
    elif xunit=='eV':
        hc_over_x  = hc_over_eV
    else:       
        assert 0, "unexpected xunit %s " % xunit


    try:
        e_nm = hc_over_x/energy  
    except RuntimeWarning:
        e_nm = float('inf')      
        log.warn("RuntimeWarning in division for %s " % repr(s)) 

    vv = np.column_stack([e_nm,val])
    return vv
    

def read_properties( xmlnode ):
    data = {}       
    for matrix in xmlnode.findall(tag("matrix")):
        xref = matrix.attrib['name']
        assert matrix.attrib['coldim'] == '2' 
        data[xref] = as_optical_property_vector( matrix.text )
    pass
    properties = {} 
    for property_ in xmlnode.findall(tag("property")):
        prop = property_.attrib['name']
        xref = property_.attrib['ref']
        #assert xref in data   # failing for LXe 
        if xref in data:
            properties[prop] = data[xref] 
        else:
            log.warn("xref not in data for property_ %s " % repr(property_))
        pass
    return properties


class MaterialProperties(DaeObject):
    def __init__(self, properties, xmlnode):
        self.properties = properties
        self.xmlnode = xmlnode
    @staticmethod
    def load(collada, localscope, xmlnode):
        properties = read_properties(xmlnode)
        return MaterialProperties( properties, xmlnode ) 
    def __repr__(self):
        return "<MaterialProperties keys=%s >" % (str(self.properties.keys())) 


class OpticalSurface(DaeObject):
    @classmethod
    def lookup(cls, localscope, surfaceproperty):
        assert surfaceproperty in localscope['surfaceproperty'], localscope
        return localscope['surfaceproperty'][surfaceproperty]

    def __init__(self, name=None, finish=None, model=None, type_=None, value=None, properties=None, xmlnode=None):
        """
        Reference

        * `materials/include/G4OpticalSurface.hh`

        """
        self.name = name
        self.finish = finish # 0:polished (smooth perfectly polished surface) 3:ground (rough surface)   (mostly "ground", ESR interfaces "polished")
        self.model = model   # 0:glisur  1:UNIFIED  2:LUT   (all are UNIFIED)
        self.type_ = type_   # 0:dielectric_metal 1:dielectric_dielectric     (all are dielectric_metal)
        self.value = value   # 1. 0. (ESRAir) or 0.2 (Pool Curtain/Liner)
        self.properties = properties
        self.xmlnode = xmlnode

    @staticmethod 
    def load(collada, localscope, xmlnode):
        name = xmlnode.attrib['name'] 
        finish = xmlnode.attrib['finish'] 
        model = xmlnode.attrib['model'] 
        type_ = xmlnode.attrib['type'] 
        value = xmlnode.attrib['value'] 
        properties = read_properties(xmlnode)
        return OpticalSurface(name, finish, model, type_, value, properties, xmlnode )

    def __repr__(self):
        return "<OpticalSurface f%s m%s t%s v%s p%s >" % (self.finish,self.model,self.type_,self.value,str(",".join(["%s:%s" % (k,len(self.properties[k])) for k in self.properties]))) 
        #return "%s" % (str(",".join(self.properties.keys()))) 



class SkinSurface(DaeObject):
    """
    skinsurface/volumeref/@ref are LV names

    ::

        dump_skinsurface
        [00] <SkinSurface PoolDetails__NearPoolSurfaces__NearPoolCoverSurface RINDEX,REFLECTIVITY >
             PoolDetails__lvNearTopCover0xad9a470
        [01] <SkinSurface AdDetails__AdSurfacesAll__RSOilSurface BACKSCATTERCONSTANT,SPECULARSPIKECONSTANT,REFLECTIVITY,SPECULARLOBECONSTANT >
             AdDetails__lvRadialShieldUnit0xaea9f58
        [02] <SkinSurface AdDetails__AdSurfacesAll__AdCableTraySurface RINDEX,REFLECTIVITY >
             AdDetails__lvAdVertiCableTray0xaf28da0
        [03] <SkinSurface PoolDetails__PoolSurfacesAll__PmtMtTopRingSurface RINDEX,REFLECTIVITY >
             PMT__lvPmtTopRing0xaf434c8
        [04] <SkinSurface PoolDetails__PoolSurfacesAll__PmtMtBaseRingSurface RINDEX,REFLECTIVITY >
             PMT__lvPmtBaseRing0xaf43520

    """
    def __init__(self, name=None, surfaceproperty=None, volumeref=None, xmlnode=None):
        self.name = name
        self.surfaceproperty = surfaceproperty
        self.volumeref = volumeref
        self.xmlnode = xmlnode
        self.debug = True
    @staticmethod
    def load(collada, localscope, xmlnode):
        name = xmlnode.attrib['name']
        surfaceproperty = xmlnode.attrib['surfaceproperty']
        surfaceproperty = OpticalSurface.lookup( localscope, surfaceproperty)
        volumeref = xmlnode.find(tag('volumeref'))
        assert volumeref is not None
        volumeref = volumeref.attrib['ref']     
        return SkinSurface(name, surfaceproperty, volumeref, xmlnode)
    def __repr__(self):
        elide = "__dd__Geometry__"
        name = str(self.name)
        if name.startswith(elide):
            name = name[len(elide):]
        smry = "<SkinSurface %s %s >" % (name, str(self.surfaceproperty)) 
        if self.debug:
            lvr = self.volumeref
            if lvr.startswith(elide):
                lvr=lvr[len(elide):]
            lvn = DAENode.lvfind(self.volumeref)  # lvfind lookup forces the parsing order
            smry += "\n" + "     %s %s " % (lvr, len(lvn))
        return smry

class BorderSurface(DaeObject):
    """ 
    """
    def __init__(self, name=None, surfaceproperty=None, physvolref1=None, physvolref2=None, xmlnode=None):
        self.name = name
        self.surfaceproperty = surfaceproperty
        self.physvolref1 = physvolref1
        self.physvolref2 = physvolref2
        self.xmlnode = xmlnode
        self.debug = True
    @staticmethod
    def load(collada, localscope, xmlnode):
        name = xmlnode.attrib['name']
        surfaceproperty = xmlnode.attrib['surfaceproperty']
        surfaceproperty = OpticalSurface.lookup( localscope, surfaceproperty)
        physvolref = xmlnode.findall(tag('physvolref'))
        assert len(physvolref) == 2
        physvolref1 = physvolref[0].attrib['ref']     
        physvolref2 = physvolref[1].attrib['ref']     
        return BorderSurface(name, surfaceproperty, physvolref1, physvolref2, xmlnode)
    def __repr__(self):
        def elide_(s): 
            elide = "__dd__Geometry__"
            if s.startswith(elide):
                return s[len(elide):]
            return s
        nlin = "\n     "
        smry = "<BorderSurface %s %s >" % (elide_(self.name), str(self.surfaceproperty)) 
        if self.debug:
            pvr1 = elide_(self.physvolref1)
            pv1 = DAENode.pvfind(self.physvolref1)  # pvfind lookup forces the parsing order
            hdr1 = "pv1 (%s) %s " % (len(pv1), pvr1)
            smry += nlin + nlin.join([hdr1]+map(str,pv1))

            pvr2 = elide_(self.physvolref2)
            pv2 = DAENode.pvfind(self.physvolref2)
            hdr2 = "pv2 (%s) %s " % (len(pv2), pvr2)
            smry += nlin + nlin.join([hdr2]+map(str,pv2))
        return smry


class VolMap(dict):
    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa)


class DAEExtra(DaeObject):
    """

    Non-distributed extra nodes are conventrated at 

    ::

                <library_nodes>
                   <node.../> 
                   <node.../> 
                   <node.../> 
                   <extra>
                      <opticalsurface.../>
                      <skinsurface.../>
                      <bordersurface.../>
                      <meta>
                         <bsurf.../>   # ? debug only ?
                      </meta>
                   </extra>
                </library_nodes>

    ::

        066057   <library_nodes>
        066058     <node id="__dd__Geometry__PoolDetails__lvNearTopCover0xc137060">
        066059       <instance_geometry url="#near_top_cover_box0xc23f970">
        066060         <bind_material>
        066061           <technique_common>
        066062             <instance_material symbol="PPE" target="#__dd__Materials__PPE0xc12f008"/>
        066063           </technique_common>
        066064         </bind_material>
        066065       </instance_geometry>
        066066     </node>
        ... 
        152905     <node id="World0xc15cfc0">
        152906       <instance_geometry url="#WorldBox0xc15cf40">
        152907         <bind_material>
        152908           <technique_common>
        152909             <instance_material symbol="Vacuum" target="#__dd__Materials__Vacuum0xbf9fcc0"/>
        152910           </technique_common>
        152911         </bind_material>
        152912       </instance_geometry>
        152913       <node id="__dd__Structure__Sites__db-rock0xc15d358">
        152914         <matrix>
        152915                 -0.543174 -0.83962 0 -16520
        152916 0.83962 -0.543174 0 -802110
        152917 0 0 1 -2110
        152918 0.0 0.0 0.0 1.0
        152919 </matrix>
        152920         <instance_node url="#__dd__Geometry__Sites__lvNearSiteRock0xc030350"/>
        152921         <extra>
        152922           <meta id="/dd/Structure/Sites/db-rock0xc15d358">
        152923             <copyNo>1000</copyNo>
        152924             <ModuleName></ModuleName>
        152925           </meta>
        152926         </extra>
        152927       </node>
        152928     </node>
        152929     <extra>
        152930       <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" type="0" value="1">
        152931         <matrix coldim="2" name="REFLECTIVITY0xc04f6a8">1.5e-06 0 6.5e-06 0</matrix>
        152932         <property name="REFLECTIVITY" ref="REFLECTIVITY0xc04f6a8"/>
        152933         <matrix coldim="2" name="RINDEX0xc33da70">1.5e-06 0 6.5e-06 0</matrix>
        152934         <property name="RINDEX" ref="RINDEX0xc33da70"/>
        152935       </opticalsurface>
        ...
        153188       <skinsurface name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface">
        153189         <volumeref ref="__dd__Geometry__PoolDetails__lvNearTopCover0xc137060"/>
        153190       </skinsurface> 
        ...
        153290       <bordersurface name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop">
        153291         <physvolref ref="__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468"/>
        153292         <physvolref ref="__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0"/>
        153293       </bordersurface>
        ...
        153322       <meta>
        153323         <bsurf name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop">
        153324           <pv copyNo="1000" name="__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap" ref="__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468"/>
        153325           <pv copyNo="1000" name="__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR" ref="__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0"/>
        153326         </bsurf>
        ...
        153359       </meta>
        153360     </extra>
        153361   </library_nodes>




    """
    def __init__(self, opticalsurface=None, skinsurface=None, bordersurface=None, skinmap=None, bordermap=None, xmlnode=None):
        self.opticalsurface = opticalsurface
        self.skinsurface = skinsurface
        self.bordersurface = bordersurface
        self.skinmap = skinmap
        self.bordermap = bordermap

    @staticmethod 
    def load(collada, localscope, xmlnode):
        if 'surfaceproperty' not in localscope:
            localscope['surfaceproperty'] = {} 

        opticalsurface = []
        for elem in xmlnode.findall(tag("opticalsurface")):
            surf = OpticalSurface.load(collada, localscope, elem)
            localscope['surfaceproperty'][surf.name] = surf
            opticalsurface.append(surf)
        log.debug("loaded %s opticalsurface " % len(opticalsurface))

        skinmap = {}
        skinsurface = []
        for elem in xmlnode.findall(tag("skinsurface")):
            skin = SkinSurface.load(collada, localscope, elem)
            skinsurface.append(skin)

            if skin.volumeref not in skinmap:
                skinmap[skin.volumeref] = []
            pass
            skinmap[skin.volumeref].append(skin)

        log.debug("loaded %s skinsurface " % len(skinsurface))

        bordermap = {}
        bordersurface = []
        for elem in xmlnode.findall(tag("bordersurface")):
            bord = BorderSurface.load(collada, localscope, elem)
            bordersurface.append(bord)

            if bord.physvolref1 not in bordermap:
                bordermap[bord.physvolref1] = []
            bordermap[bord.physvolref1].append(bord)

            if bord.physvolref2 not in bordermap:
                bordermap[bord.physvolref2] = []
            bordermap[bord.physvolref2].append(bord)

        log.debug("loaded %s bordersurface " % len(bordersurface))

        pass
        return DAEExtra(opticalsurface, skinsurface, bordersurface, skinmap, bordermap, xmlnode)

    def __repr__(self):
        return "%s skinsurface %s bordersurface %s opticalsurface %s skinmap %s bordermap %s " % (self.__class__.__name__, 
             len(self.skinsurface),len(self.bordersurface),len(self.opticalsurface), len(self.skinmap),len(self.bordermap)) 
 
    def dump_skinsurface(self):
        print "dump_skinsurface" 
        print "\n".join(map(lambda kv:"[%-0.2d] %s" % (kv[0],str(kv[1])),enumerate(self.skinsurface))) 
    def dump_bordersurface(self):
        print "dump_bordersurface" 
        print "\n".join(map(lambda kv:"[%-0.2d] %s" % (kv[0],str(kv[1])),enumerate(self.bordersurface))) 
    def dump_opticalsurface(self):
        print "\n".join(map(str,self.opticalsurface)) 

    def dump_skinmap(self):
        print "dump_skinmap" 
        for iv,(v,ss) in enumerate(self.skinmap.items()):
            print 
            print iv, len(ss), v
            for j, s in enumerate(ss):
                print "   ", j, s
            pass 
        print self

    def dump_bordermap(self):
        print "dump_bordermap" 
        for iv,(v,ss) in enumerate(self.bordermap.items()):
            print 
            print iv, len(ss), v
            for j, s in enumerate(ss):
                print "   ", j, s
            pass 
        print self






class DAESubTree(list):
    """
    Recursively creates a list-of-strings representation 
    of a tree structure. Requires the nodes to have a children
    attribute which lists other nodes.
    """
    def __init__(self, top, maxdepth=-1, text=True, maxsibling = 5):
        list.__init__(self)
        self.maxdepth = maxdepth
        self.text = text
        self.cut = maxsibling 
        self( top )

    __str__ = lambda _:"\n".join(_)

    def __call__(self, node, depth=0, sibdex=-1, nsibling=-1 ):
        """
        Setting the node to a string acts to stop the recursion
        """
        if not hasattr(node,'children'):
            nchildren = 0  
        else:
            nchildren = len(node.children) 
        pass
        elided = type(node) == str
        indent = "   " * depth     # done here as difficult to do in a webpy template
        if self.text:
            if elided:
                obj = "..."
            else:
                nodelabel = "%-2d %-5d %-3d" % (depth, node.index, nchildren )
                obj = "[%s] %s %3d/%3d : %s " % (nodelabel, indent, sibdex, nsibling, node)
        else:
            obj = (node, depth, sibdex, indent)
        pass     
        self.append( obj )


        if nchildren == 0:# leaf
            pass
        else:
            if depth == self.maxdepth:
                pass
            else:    
                shorten = nchildren > self.cut*2    
                for sibdex, child in enumerate(node.children):
                    if shorten:
                        if sibdex < self.cut or sibdex > nchildren - self.cut:
                            pass
                        elif sibdex == self.cut:    
                            child = "..."
                        else:
                            continue
                    pass         
                    self(child, depth + 1, sibdex, nchildren )





class DAECopy(object):
    """
    Non-Node objects, ie Effect, Material, Geometry have clearly defined places 
    to go within the `library_` elements and there is no need to place other
    elements inside those.

    The situation is not so clear with  the MaterialNode, GeometryNode, NodeNode, Node
    which live in a containment heirarcy, and for Node can contain others inside them.
    """
    def __init__(self, top, opts ):

        self.opts = opts 

        dae = collada.Collada()
        dae.assetInfo.upaxis = 'Z_UP'    # default is Y_UP

        self.dae = dae
        self.maxdepth = int(self.opts.get('maxdepth',0))   # default to no-recursion

        cpvtop = self( top )    # recursive copier
        self.cpvtop = cpvtop

        content = [cpvtop]
        if 'extra' in opts:
            cextra = collada.scene.ExtraNode( opts['extra']  )
            content.append(cextra) 


        cscene = collada.scene.Scene("DefaultScene", content )
        self.dae.scenes.append(cscene)
        self.dae.scene = cscene
        self.dae.save()             #  the save takes ~60% of total CPU time


    def load_effect( self, effect ):
        """
        :param effect: to be copied  

        Creates an effect from the xmlnode of an old one into 
        the new collada document being created
        """
        #ceffect = collada.material.Effect.load( self.dae, {},  effect.xmlnode ) 
        #ceffect = copy.copy( effect )
        ceffect = effect 
        ceffect.double_sided = True  
        self.dae.effects.append(ceffect)  # pycollada managed not adding duplicates 
        return ceffect

    def load_material( self, material ):    
        """
        :param material:

        must append the effect before can load the material that refers to it 
        """
        #cmaterial = collada.material.Material.load( self.dae, {} , material.xmlnode )
        #cmaterial = copy.copy( material )
        cmaterial = material
        cmaterial.double_sided = True   
        self.dae.materials.append(cmaterial)
        return cmaterial

    def load_geometry( self, geometry  ):
        """
        :param geometry:

        Profiling points to this consuming half the time
        attempts to use a deepcopy instead lead to 

        ::

             File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/copy.py", line 189, in deepcopy
                 y = _reconstruct(x, rv, 1, memo)
             File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/copy.py", line 329, in _reconstruct
                 y.append(item)
             File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/pycollada-0.4-py2.6.egg/collada/util.py", line 226, in append
                 self._addindex(obj)
             File "/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/pycollada-0.4-py2.6.egg/collada/util.py", line 152, in _addindex
                 _idx = self._index
             AttributeError: 'IndexedList' object has no attribute '_index'

        """
        #cgeometry = collada.geometry.Geometry.load( self.dae, {}, geometry.xmlnode)   # this consumes 43% of time
        #cgeometry = copy.deepcopy( geometry )
        #cgeometry = copy.copy( geometry )
        cgeometry = geometry

        self.dae.geometries.append(cgeometry)
        return cgeometry
 
    def copy_geometry_node( self, geonode ):
        """
        ::

            <instance_geometry url="#RPCStrip0x886a088">
               <bind_material>
                  <technique_common>
                      <instance_material symbol="WHITE" target="#__dd__Materials__MixGas0x8837740"/>
                 </technique_common>
               </bind_material>
            </instance_geometry>
        """
        cgeometry = self.load_geometry( geonode.geometry )
        cmaterials = []    # actually matnodes
        for matnode in geonode.materials:
            material = matnode.target
            ceffect = self.load_effect( material.effect )
            cmaterial = self.load_material( material )
            # cmaterial.double_sided = True   # geo node keeps a reference to the material, so changing this doesnt help ?
            cmatnode = collada.scene.MaterialNode( matnode.symbol, cmaterial, matnode.inputs )
            cmaterials.append(cmatnode)
        pass     
        cgeonode = collada.scene.GeometryNode( cgeometry, cmaterials )
        return cgeonode


    def faux_copy_geometry_node(self, geonode):
        """
        Not really copying, just borrowing objects owned from the original .dae into the subcopy one
        """
        self.dae.geometries.append( geonode.geometry )
        for matnode in geonode.materials:
            material = matnode.target
            self.dae.effects.append( material.effect ) 
            self.dae.materials.append( material ) 
        pass    
        return geonode

    def make_id(self, bid ):
        if self.opts.get('blender',False):
            return bid[-8:]     # just the ptr, eg xa8bffe8 as blender is long id challenged
        return bid 

    def __call__(self, vnode, depth=0, index=0 ):
        """
        The translation of the Geant4 model into Collada being used has:

        * LV nodes contain instance_geometry and 0 or more node(PV)  elements  
        * PV nodes contain matrix and instance_node (pointing to an LV node) **ONLY**
          they are merely placements within their holding LV node. 
          
        DAENode are created by collada raw nodes traverse hitting leaves, ie
        with recursion node path  Node/NodeNode/GeometryNode or xml structure
        node/instance_node/instance_geometry 

        Thus DAENode instances correspond to::
        
             containing PV
                instance_node referenced LV
                    LV referenced geometry 

        NB this means the PV and LV are referring to different volumes, the PV
        being the containing parent PV. Because of this it is incorrect to recurse
        on the PV, its only the LV that (maybe) holds child PV that require to
        be recursed to.   Stating another way, the PV is the containing parent volume
        so its just plain wrong to recurse on it.

        NodeNode children are those of the referred to node

        :: 

             63868     <node id="__dd__Geometry__RPC__lvRPCGasgap140xa8c0268">
             63869       <instance_geometry url="#RPCGasgap140x886a0f0">
             63870         <bind_material>
             63871           <technique_common>
             63872             <instance_material symbol="WHITE" target="#__dd__Materials__Air0x8838278"/>
             63873           </technique_common>
             63874         </bind_material>
             63875       </instance_geometry>
             63876       <node id="__dd__Geometry__RPC__lvRPCGasgap14--pvStrip14Array--pvStrip14ArrayOne..1--pvStrip14Unit0xa8c02c0">
             63877         <matrix>
             63878                 6.12303e-17 1 0 -910
             63879                 -1 6.12303e-17 0 0
             63880                  0 0 1 0
             63881                  0.0 0.0 0.0 1.0
             63882         </matrix>
             63883         <instance_node url="#__dd__Geometry__RPC__lvRPCStrip0xa8c01d8"/>
             63884       </node>


        Schematically::

              <node id="lvname1" >         
                 <instance_geometry url="#geo1" ... />
                 <node id="pvname1" >   "PV"  
                    <matrix/>
                    <instance_node url="#lvname2" >  "LV"
                        
                         metaphorically the instance node passes 
                         thru to the referred to node for the raw collada recurse
                         and makes that node element "invisble"
                         (not appearing in the nodepath used to create the DAENode)
                         hence  Node/NodeNode/GeometryNode
                                 pv     lv        geo      <<< SO PV is the parent of the LV, not the same volume ???

                           <node id="lvname2">      "LV"
                               <instance_geometry url="#geo2" />   "GEO"

                               <node id="pvname3" >
                                    <matrix/>
                                    <instance_node url="#lvname4" />
                               </node>
                               ...
                           </node> 

                    </instance_node>
                 </node>
                 <node id="pvname2" >            
                    <matrix/>
                    <instance_node url="#lvname3" />
                 </node>
                 ...
              </node>

        """
        #log.debug( "    " * depth + "[%d.%d] %s " % (depth, index, daenode))
        pvnode, lvnode, geonode = vnode.pv, vnode.lv, vnode.geo
        # NB the lvnode is a NodeNode instance


        # copy the instance_geometry node referred to by the LV  
        cnodes = []
        cgeonode = self.copy_geometry_node( geonode )
        #cgeonode = self.faux_copy_geometry_node( geonode )
        cnodes.append(cgeonode)  

        # collect children of the referred to LV, ie the contained PV
        if not hasattr(vnode,'children') or len(vnode.children) == 0:# leaf
            pass
        else:
            if depth == self.maxdepth:  # stop the recursion when hit maxdepth
                pass
            else:
                for index, child in enumerate(vnode.children):  
                    cnode = self(child, depth + 1, index )       ####### THE RECURSIVE CALL ##########
                    cnodes.append(cnode)
            pass

        # bring together the LV copy , NB the lv a  NodeNode instance, hence the `.node` referral in the below 
        # (properties hide this referral for id/children but not for transforms : being explicit below for clarity )
        copy_ = lambda _:_.load(collada, _.xmlnode)     # create a collada object from the xmlnode representation of another
        clvnode = collada.scene.Node( self.make_id(lvnode.node.id) , children=cnodes, transforms=map(copy_,lvnode.node.transforms) ) 

        # unlike the other library_ pycollada does not prevent library_nodes/node duplication 
        if not clvnode.id in self.dae.nodes:
            self.dae.nodes.append(clvnode)
        
        # deal with the containing/parent PV, that references the above LV  
        refnode = self.dae.nodes[clvnode.id]  
        cnodenode = collada.scene.NodeNode( refnode ) 
        cpvnode = collada.scene.Node( self.make_id(pvnode.id) , children=[cnodenode], transforms=map(copy_,pvnode.transforms) ) 

        #log.debug("cpvnode %s " % tostring_(cpvnode) )
        return cpvnode


    def __str__(self):
        """
        Even after popping the top node id blender still complains::

            cannot find Object for Node with id=""
            cannot find Object for Node with id=""
            cannot find Object for Node with id=""
            cannot find Object for Node with id=""

        """
        #self.dae.save()   this stay save was almost doubling CPU time 

        if self.opts.get('blender',False):
            if self.cpvtop.xmlnode.attrib.has_key('id'):
                topid = self.cpvtop.xmlnode.attrib.pop('id')
                log.info("popped the top for blender %s " % topid )

        out = StringIO()
        writeXML(self.dae.xmlnode, out )
        return out.getvalue()



def getSubCollada(arg, cfg ):
    """
    DAENode kinda merges LV and PV, but this should be a definite place, so regard as PV

    :return: collada XML string for sub geometry
    """
    arg, maxdepth = DAENode.interpret_arg(arg)
    cfg['maxdepth'] = maxdepth
    log.info("getSubCollada arg maxdepth handling %s %s " % (arg, maxdepth))

    indices = DAENode.interpret_ids(arg)
    assert len(indices) == 1 
    index = indices[0]
    log.info("geom subcopy arg %s => index %s cfg %s " % (arg, index, cfg) )
    top = DAENode.indexget(index)  


    extra = E.extra()
    meta = E.meta("subcopy arg %s index %s maxdepth %s " % (arg, index, cfg['maxdepth']) )
    extra.append(meta)
    for a in reversed(top.ancestors()):
        ancestor = E.ancestor(str(a.index), id=a.id)
        extra.append(ancestor)
    pass
    extra.append(E.subroot(str(top.index), id=top.id))
    pass
    for c in top.children:
        child = E.child(str(c.index), id=c.id)
        extra.append(child)
    pass
    cfg['extra'] = extra

    vc = DAECopy(top, cfg )
    svc = str(vc)

    subpath = cfg.get('subpath', None)
    if not subpath is None and cfg.get('daesave',False) == True:
        log.info("daesave to %s " % subpath )
        fp = open(subpath, "w") 
        fp.write(svc)
        fp.close()

    return svc

def node(arg, cfg ):
    """
    Present info for a single node
    """
    arg, maxdepth = DAENode.interpret_arg(arg)
    cfg['maxdepth'] = maxdepth

    ancestors = cfg.get('ancestors', cfg.get('a',None))
    children  = cfg.get('children', cfg.get('c',None))
    geometry = cfg.get('geometry', cfg.get('g',None))

    ids = DAENode.interpret_ids(arg)
    hdr = ["_dump [%s] => ids %s " % (arg, str(ids) ), "cfg %s " % cfg, "" ]
 
    vnode_ = lambda _:DAENode.registry[_] 
    nodes = map(vnode_, ids )

    out = []
    for node in nodes:
        out.append(node)
        if geometry:
            out.extend( present_geometry(node.boundgeom))
        if ancestors:
            for a in node.ancestors():
                out.insert(0,"a %s" % a) 
        if children:
            for c in node.children:
                out.append("c %s" % c) 
    pass            
    return "\n".join(map(str,hdr+out))

def getTextTree(arg, cfg):
    """
    Present a text tree of the volume heirarchy from the root(s) defined 
    by the argument. 
    """
    arg, maxdepth = DAENode.interpret_arg(arg)
    cfg['maxdepth'] = maxdepth

    ancestors = cfg.get('ancestors', cfg.get('a',None))
    indices = DAENode.interpret_ids(arg)
    assert len(indices) == 1 , "currently restricting to single roots "
    index = indices[0]

    node = DAENode.indexget(index)
    anc = []
    if ancestors:
        for a in node.ancestors():
            anc.insert(0,"a %s" % a)

    tre = DAESubTree(node, maxdepth=int(cfg.get('maxdepth','-1')))
    return "\n".join(map(str,anc+tre+[""])) 




class Defaults(object):
    logformat = "%(asctime)s %(name)-20s:%(lineno)-3d %(levelname)-8s %(message)s"
    loglevel = "INFO"
    logpath = None
    daepath = "dyb"
    daedbpath = None
    webserver = False
    tree = False
    node = False
    geom = False
    daesave = False
    blender = False
    ancestors = "YES"
    geometry = "YES"
    subpath = "subcopy.dae"
    maxdepth = -1
    points = True
    faces = True
    insertsize = 0
    ipy = False
    surface = False


def resolve_path(path_):
    pvar = "_".join(filter(None,["DAE_NAME",path_,]))
    pvar = pvar.upper()
    path = os.environ.get(pvar,None)
    log.info("Using pvar %s to resolve path : %s " % (pvar, path) )
    assert not path is None, "Need to define envvar pointing to geometry file"
    assert os.path.exists(path), path
    return path


def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=defopts.logpath , help="logging path" )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG. Default %default"  )
    op.add_option("-f", "--logformat", default=defopts.logformat , help="logging format" )

    op.add_option("-p", "--daepath", default=defopts.daepath , help="Path to the original geometry file. Default %default ")
    op.add_option(      "--daedbpath", default=defopts.daedbpath , help="Path to the summary SQLite DB, when None use daepath with '.db' appended. Default %default ")
    op.add_option(      "--ipy",  action="store_true", default=defopts.ipy , help="Drop into embedded ipython. Default %default ")

    # three way split 
    op.add_option("-d", "--node", action="store_true", default=defopts.node , help="Text representation of a single volume. Default %default." )
    op.add_option("-t", "--tree", action="store_true", default=defopts.tree , help="Text representation of the tree from the target volume. Default %default.")
    op.add_option("-s", "--geom",  action="store_true", default=defopts.geom, help="Perform a recursive subcopy of the geometry. Default %default. ")

    op.add_option("-a", "--ancestors", default=defopts.ancestors , help="Include ancestor nodes in the text dumps. Default %default.")
    op.add_option("-g", "--geometry", default=defopts.geometry ,  help="Include geometry details in the text dumps. Default %default.")

    op.add_option("-w", "--webserver", action="store_true", default=defopts.webserver, help="Start a webserver on local node. Default %default." )

    op.add_option("-e", "--daesave", action="store_true",  default=defopts.daesave , help="Save the subgeometry to a file. Default %default." )
    op.add_option("-O", "--subpath", default=defopts.subpath , help="Path in which to save subgeometry, when `-e/--daesave` option is used. Default %default." )
    op.add_option("-x", "--maxdepth", type=int, default=defopts.maxdepth, help="Restrict the tree depth of the copy, -1 for full tree from the specified root volume. Default %default " )
    op.add_option("-b", "--blender",  action="store_true", default=defopts.blender , help="Change some aspects of exported geometry for blender compatibility. Default %default. ")
    op.add_option("-P", "--nopoints",  dest="points", action="store_false", default=defopts.points , help="Prevent the timeconsuming persisting all points. Default %default. ")
    op.add_option("-F", "--nofaces",   dest="faces", action="store_false", default=defopts.faces , help="Prevent the timeconsuming persisting all faces. Default %default. ")
    op.add_option("-i", "--insertsize", type="int", default=defopts.insertsize, help="Control chunk size of DB inserts, zero for all at the end. Default %default. ")
    op.add_option(       "--surface", action="store_true", default=defopts.surface, help="Surface checking. Default %default. ")

    opts, args = op.parse_args()
    del sys.argv[1:]   # avoid confusing webpy with the arguments

    level = getattr( logging, opts.loglevel.upper() )

    if opts.logpath:  # logs to file as well as console, needs py2.4 + (?)
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(opts.logformat)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)  # add the handler to the root logger
    else:
        logging.basicConfig(format=opts.logformat,level=level)
    pass
    log.info(" ".join(sys.argv))

    #daepath = os.path.expandvars(os.path.expanduser(opts.daepath))
    daepath = resolve_path(opts.daepath)
    if not daepath[0] == '/':
        daepath = os.path.abspath(daepath)
    assert os.path.exists(daepath), (daepath,"DAE file not at the new expected location, please create the directory and move the .dae  there, please")
    opts.daepath = daepath
    if opts.daedbpath is None:
        opts.daedbpath = opts.daepath + '.db'

    return opts, args



# webpy interface glue
class _index:
    def GET(self):
        return "\n".join(["_index %s " % len(DAENode.registry), __doc__ ])
class _node:
    def GET(self, arg):
        return node(arg, dict(web.input().items()))
class _tree:
    def GET(self, arg):
        return tree(arg, dict(web.input().items()))
class _geom:
    def GET(self, arg):
        log.info("_geom.GET %s " % arg )
        return getSubCollada(arg, dict(web.input().items()))

def main():
    opts, args = parse_args(__doc__) 
    DAENode.parse( opts.daepath )
    if opts.webserver:
        webserver()
    elif opts.node:
        print node(args[0])
    elif opts.tree:
        print getTextTree(args[0], vars(opts))
    elif opts.geom:
        print getSubCollada(args[0], vars(opts))

    if opts.ipy:
        from daecommon import splitname, shortname, fromjson
        bordersurface = dict((splitname(_.name)[1],_) for _ in DAENode.extra.bordersurface)
        log.info("droping into IPython, try:\n%s\n" % examples ) 
        import IPython
        IPython.embed()
    pass

examples = r"""

b=bordersurface['SSTOil'] 
b?? 

"""


def webserver():
    log.info("starting webserver ")
    urls = ( 
             '/',           '_index', 
             '/node/(.+)?', '_node', 
             '/tree/(.+)?', '_tree', 
             '/geom/(.+)?', '_geom', 
           )
    app = web.application(urls, globals())
    app.run() 


if __name__ == '__main__':
    main()


