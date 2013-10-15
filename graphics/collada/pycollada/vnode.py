#!/usr/bin/env python
"""
Attempt to create the VNode heirarcy out of a raw collada traverse 
without monkeying around.

::

    In [21]: len(VNode.registry)
    Out[21]: 12230

    In [22]: len(boundgeom)
    Out[22]: 12230

    In [23]: boundgeom[1000]
    Out[23]: <BoundGeometry id=RPCStrip0x92ed088, 1 primitives>

    In [25]: VNode.registry[1000].pv 
    Out[25]: '__dd__Geometry__RPC__lvRPCGasgap23--pvStrip23Array--pvStrip23ArrayOne..6--pvStrip23Unit0xb3445c8'

    In [26]: VNode.registry[1000].lv
    Out[26]: '__dd__Geometry__RPC__lvRPCStrip0xb3431d8'

    In [27]: VNode.registry[1000].geo
    Out[27]: 'RPCStrip0x92ed088'



"""
import pickle
import os, logging, hashlib
log = logging.getLogger(__name__)


class VNode(object):
    registry = []
    lookup = {}
    ids = set()
    created = 0
    root = None
    pkpath = "vnode.pk"
    rawcount = 0

    @classmethod
    def recurse(cls, node , ancestors=[], limit=None ):
        """
        This recursively visits 12230*3 = 36690 Nodes.  
        The below pattern of triplets of node types is followed precisely, due to 
        the node/instance_node/instance_geometry layout adopted for the dae file.

        The triplets are collected into VNode on every 3rd leaf node.

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
        if cls.rawcount < 10:
            log.info("recurse [%s] %s : %s " % (cls.rawcount, id(node), node ))
        if not limit is None and cls.rawcount > limit:
            log.warn("truncating recurse at rawcount %s " % cls.rawcount )
            return

        if not hasattr(node,'children') or len(node.children) == 0:# leaf
            cls.make( ancestors + [node])
        else:
            for child in node.children:
                cls.recurse(child, ancestors = ancestors + [node], limit = limit )

    @classmethod
    def summary(cls):
        log.info("rawcount %s " % cls.rawcount )

    @classmethod
    def save(cls):
        log.info("saving to %s " % cls.pkpath )
        pickle.dump( cls.registry, open( cls.pkpath, "wb" ) ) 

    @classmethod
    def load(cls):
        log.info("loading from %s " % cls.pkpath )
        cls.registry = pickle.load( open( cls.pkpath, "rb" ) ) 
        for v in cls.registry:
            if v.index == 0:
                cls.root = v
                break

    @classmethod
    def find_uid(cls, bid, decodeNCName=True):
        """
        :param bid: basis ID

        Find a unique id for the emerging VNode
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
        node = cls(nodepath)
        if node.index == 0:
            cls.root = node

        cls.registry.append(node)
        # digest keyed lookup gives fast access to node parents
        # the digest represents a path through the tree of nodes 
        cls.lookup[node.digest] = node   
        cls.created += 1

        parent = cls.lookup.get(node.parent_digest)
        node.parent = parent
        if parent is None:
            log.warn("failed to find parent for %s " % node )
        else:
            parent.children.append(node)  

        if cls.created % 1000 == 0:
            log.info("make %s : [%s] %s " % ( cls.created, id(node), node ))
        return node


    def has_parent(self, other):
        """
        Check if the other `VNode` is the parent of this one
        """
        if other.leafdepth != self.rootdepth:
            return False
        else:
            return other.digest == self.parent_digest

    @classmethod
    def walk(cls, node=None, vdepth=0):
        if node is None:
            cls.wcount = 0
            node=cls.root

        cls.wcount += 1 
        if cls.wcount % 100 == 0:
            log.info("walk %s %s %s " % ( cls.wcount, vdepth, node ))
            if hasattr(node,'boundgeom'):
                print node.boundgeom
        for subnode in node.children:
            cls.walk(subnode, vdepth+1)
                
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
        up with for the boundgeom must match the VNode ordering

        The geometry id comparison performed is a necessary condition, 
        but it does not imply correctness of the cross referencing due
        to a lot of id recycling.
        """
        log.info("index linking VNode with boundgeom %s volumes " % len(boundgeom)) 
        assert len(cls.registry) == len(boundgeom)
        for vn,bg in zip(VNode.registry,boundgeom):
            vn.boundgeom = bg
            bg.vnode = vn
            assert vn.geo == bg.original.id   
        log.info("index linking completed")    


    def ancestors(self):
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
        p = self.parent
        while p is not None:
            yield p
            p = p.parent

    def __init__(self, nodepath):
        """
        :param nodepath: list of node instances identifying all ancestors and the leaf geometry node
        :param rootdepth: depth 
        :param leafdepth: 

        Currently `rootdepth == leafdepth - 2`,  making each VNode be constructed out 
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

        # store ids to allow pickling 
        self.pv = pv.id
        self.lv = lv.id   
        self.geo = geo.geometry.id
        pass
        self.id = self.find_uid( pv.id , False)
        self.index = len(self.registry)

    def __str__(self):
        lines = []
        lines.append("VNode(%s,%s)[%s,%s]" % (self.rootdepth,self.leafdepth,self.index, self.id) )
        #lines.append("  dig:%s" % (self.digest) )
        #lines.append(" pdig:%s" % (self.parent_digest) )
        return "\n".join(lines)

    __repr__ = __str__



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    path = os.path.expandvars("$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae")
    import collada 
    log.info("pycollada parse %s " % path )
    dae = collada.Collada(path)
    log.info("pycollada parse completed ")
    boundgeom = list(dae.scene.objects('geometry'))
    top = dae.scene.nodes[0]
    log.info("pycollada binding completed, found %s  " % len(boundgeom))

    log.info("create VNode heirarchy ")
    usecache = False
    if usecache and os.path.exists(VNode.pkpath):
        VNode.load()
    else:
        VNode.recurse(top, limit=None)
        VNode.summary()
        VNode.save()
    
    VNode.indexlink( boundgeom )
    VNode.walk()



