#!/usr/bin/env python
"""
Geant4 level interpretation of G4DAEWrite exported pycollada geometry
=======================================================================

Attempt to create the DAENode heirarcy out of a raw collada traverse 
without monkeying around.


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
import collada 
from collada.xmlutil import etree as ET
from collada.xmlutil import writeXML, COLLADA_NS, E

# relying on a load signature that is not uniformly followed, works for Transforms, but too much action-at-a-distance
#collada.common.DaeObject.copy = lambda self:self.load(collada, self.xmlnode )  
#collada.scene.Transform.copy = lambda self:self.load(collada, self.xmlnode )  

# disable saves, which update the xmlnode, as the preexisting xmlnode for 
# the basis objects are being copied anyhow
collada.geometry.Geometry.save = lambda _:_
collada.material.Material.save = lambda _:_

tostring_ = lambda _:ET.tostring(getattr(_,'xmlnode'))
#from lxml.builder import E

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
        log.info("DAENode.parse pycollada parse %s " % path )
        dae = collada.Collada(path)
        log.info("pycollada parse completed ")
        boundgeom = list(dae.scene.objects('geometry'))
        top = dae.scene.nodes[0]
        log.info("pycollada binding completed, found %s  " % len(boundgeom))
        log.info("create DAENode heirarchy ")
        cls.orig = dae
        cls.recurse(top)
        cls.summary()
        cls.indexlink( boundgeom )

    @classmethod
    def recurse(cls, node , ancestors=[] ):
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

        if not hasattr(node,'children') or len(node.children) == 0:# leaf
            cls.make( ancestors + [node])
        else:
            for child in node.children:
                cls.recurse(child, ancestors = ancestors + [node] )

    @classmethod
    def indexget(cls, index):
        return cls.registry[index]

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
            maxdepth = -1
        return arg, maxdepth

    @classmethod
    def interpret_ids(cls, arg):
        """
        Interpret an arg like 0:10,400:410,300,40,top.0
        into a list of integer DAENode indices 
        """
        if "," in arg:
            args = arg.split(",")
        else:
            args = [arg]
        ids = []
        for arg in args:
            if ":" in arg:
                iarg=range(*map(int,arg.split(":")))
                ids.extend(iarg)
            else:
                if "___" in arg:
                    arg = arg.split("___")[0]   # get rid of the maxdepth indicator eg "___0"
                try:
                    int(arg)
                    ids.append(int(arg))
                except ValueError:
                    node = cls.idlookup.get(arg,None)
                    if node:
                        ids.append(node.index)
                    else:
                        log.warn("failed to lookup DAENode for arg %s " % arg)
        return ids


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
    def make(cls, nodepath ):
        """
        Creates `DAENode` instances and positions them within the volume tree
        by setting the `parent` and `children` attributes.

        A digest keyed lookup gives fast access to node parents,
        the digest represents a path through the tree of nodes.
        """
        node = cls(nodepath)
        if node.index == 0:
            cls.root = node

        cls.registry.append(node)
        cls.idlookup[node.id] = node   
        cls.lookup[node.digest] = node   
        cls.created += 1

        parent = cls.lookup.get(node.parent_digest)
        node.parent = parent
        if parent is None:
            log.warn("failed to find parent for %s (failure expected only for root node)" % node )
        else:
            parent.children.append(node)  

        if cls.created % 1000 == 0:
            log.info("make %s : [%s] %s " % ( cls.created, id(node), node ))
        return node

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
        assert len(cls.registry) == len(boundgeom)
        for vn,bg in zip(cls.registry,boundgeom):
            vn.boundgeom = bg
            bg.daenode = vn
            assert vn.geo.geometry.id == bg.original.id   
        log.info("index linking completed")    


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

    def __init__(self, nodepath):
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
        assert pv.__class__.__name__ in ('MonkeyNode','Node'), pv
        assert lv.__class__.__name__ in ('MonkeyNodeNode','NodeNode'), lv
        assert geo.__class__.__name__ in ('MonkeyGeometryNode','GeometryNode'), geo

        self.children = []
        self.leafdepth = leafdepth
        self.rootdepth = rootdepth 
        self.digest = self.md5digest( nodepath[0:leafdepth-1] )
        self.parent_digest = self.md5digest( nodepath[0:rootdepth-1] )

        # formerly stored ids rather than instances to allow pickling 
        self.pv = pv
        self.lv = lv   
        self.geo = geo
        #self.geo = geo.geometry.id
        pass
        self.id = self.find_uid( pv.id , False)
        self.index = len(self.registry)

    def matdict(self):
        if hasattr(self, '_matdict'):
            return self._matdict
        if not hasattr(self, 'boundgeom'):
            _matdict =  {}
        else:    
            bg = self.boundgeom
            msi = bg.materialnodebysymbol.items()
            assert len(msi) == 1 
            symbol, matnode= msi[0]
            matid = matnode.target.id
            _matdict = dict(matid=matid, symbol=symbol)
        self._matdict = _matdict    
        return self._matdict

    matid  = property(lambda self:self.matdict().get('matid',None))
    symbol = property(lambda self:self.matdict().get('symbol',None))

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

    def __str__(self):
        lines = []
        matdict = self.matdict()
        if self.verbosity > 0:
            lines.append("DAENode(%s,%s)[%s]    %s             %s " % (self.rootdepth,self.leafdepth,self.index, self.id, matdict.get('matid',"-") ) )
        if self.verbosity > 1:    
            lines.append("    pvid         %s " % self.pv.id )
            lines.append("    lvid         %s " % self.lv.id )
            lines.append("    ggid         %s " % self.geo.geometry.id )
        if self.verbosity > 2:    
            lines.extend(self.primitives())
        return "\n".join(lines)

    __repr__ = __str__




class DAESubTree(list):
    """
    Recursively creates a list-of-strings representation 
    of a tree structure. Requires the nodes to have a children
    attribute which lists other nodes.
    """
    cut = 5
    def __init__(self, top, maxdepth=-1, text=True):
        list.__init__(self)
        self.maxdepth = maxdepth
        self.text = text
        self( top )

    __str__ = lambda _:"\n".join(_)

    def __call__(self, node, depth=0, sibdex=0 ):
        if self.text:
            obj = "    " * depth + "[%d.%d] %s " % (depth, sibdex, node)
        else:
            if type(node) == str:
                indent = node
            else:    
                indent = "  +" * depth 
            obj = (node, depth, sibdex, indent)
        self.append( obj )
        if not hasattr(node,'children') or len(node.children) == 0:# leaf
            pass
        else:
            if depth == self.maxdepth:
                pass
            else:    
                shorten = len(node.children) > self.cut*2    
                for sibdex, child in enumerate(node.children):
                    if shorten:
                        if sibdex < self.cut or sibdex > len(node.children) - self.cut:
                            pass
                        elif sibdex == self.cut:    
                            child = "..."
                        else:
                            continue
                    pass         
                    self(child, depth + 1, sibdex)





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
        self.dae = collada.Collada()
        self.maxdepth = int(self.opts.get('maxdepth',-1))

        cpvtop = self( top )    # recursive copier
        self.cpvtop = cpvtop

        content = [cpvtop]
        if 'meta' in opts:
            cextra = collada.scene.ExtraNode( opts['meta']  )
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

    cfg['meta'] = E.extra(E.meta("subcopy arg %s index %s maxdepth %s " % (arg, index, cfg['maxdepth']) ))
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

def tree(arg, cfg):
    """
    Present a text tree of the volume heirarchy from the root(s) defined 
    by the argument. 
    """
    arg, maxdepth = DAENode.interpret_arg(arg)
    cfg['maxdepth'] = maxdepth

    ancestors = cfg.get('ancestors', cfg.get('a',None))
    indices = DAENode.interpret_ids(arg)
    assert len(indices) == 1
    index = indices[0]
    node = DAENode.indexget(index)

    anc = []
    if ancestors:
        for a in node.ancestors():
            anc.insert(0,"a %s" % a)

    tre = DAESubTree(node, maxdepth=int(cfg.get('maxdepth','-1')))
    return "\n".join(map(str,anc+tre))


class Defaults(object):
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    loglevel = "INFO"
    logpath = None
    daepath = "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae"
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

def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=defopts.logpath , help="logging path" )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG. Default %default"  )
    op.add_option("-f", "--logformat", default=defopts.logformat , help="logging format" )

    op.add_option("-p", "--daepath", default=defopts.daepath , help="Path to the original geometry file. Default %default ")

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
    daepath = os.path.expandvars(os.path.expanduser(opts.daepath))
    if not daepath[0] == '/':
        daepath = os.path.abspath(daepath)
    assert os.path.exists(daepath), (daepath,"DAE file not at the new expected location, please create the directory and move the .dae  there, please")
    opts.daepath = daepath
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
        print tree(args[0], vars(opts))
    elif opts.geom:
        print getSubCollada(args[0], vars(opts))


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


