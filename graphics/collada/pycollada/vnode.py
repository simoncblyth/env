#!/usr/bin/env python
"""
Attempt to create the VNode heirarcy out of a raw collada traverse 
without monkeying around.

"""
import pickle
import os, logging, hashlib
log = logging.getLogger(__name__)


class VNode(list):
    registry = []
    ids = set()
    created = 0
    root = None
    pkpath = "vnode.pk"

    @classmethod
    def recurse(cls, node , ancestors=[] ):
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
        if not hasattr(node,'children') or len(node.children) == 0:# leaf
            cls.make( ancestors + [node])
        else:
            for child in node.children:
                cls.recurse(child, ancestors + [node] )


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
        this = cls(nodepath)
        if this.index == 0:
            cls.root = this
        cls.registry.append(this)
        cls.created += 1

        # hook this up with its parent, apart from first node a parent should always be found  
        parent = None
        for vv in reversed(cls.registry):
            if this.has_parent(vv):  
                parent = vv
                break
        pass
        if parent is None:
            log.warn("failed to find parent for %s " % this )
        else:
            parent.append(this)  

        if cls.created % 1000 == 0:
            print cls.created, this

        return this 

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
        print "walk ", cls.wcount, vdepth, node
        for subnode in node:
            cls.walk(subnode, vdepth+1)
                
    @classmethod
    def md5digest(cls, nodepath ):
        """
        Use of id means that will change from run to run. 
        """
        dig = hashlib.md5("".join(map(lambda _:str(id(_)),nodepath))).hexdigest() 
        print dig, nodepath
        return dig

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
        list.__init__(self)
        assert len(nodepath) >= 3
        leafdepth = len(nodepath)
        rootdepth = len(nodepath) - 2

        pv, lv, geo = nodepath[-3:]
        assert pv.__class__.__name__ in ('MonkeyNode','Node'), pv
        assert lv.__class__.__name__ in ('MonkeyNodeNode','NodeNode'), lv
        assert geo.__class__.__name__ in ('MonkeyGeometryNode','GeometryNode'), geo

        self.leafdepth = leafdepth
        self.rootdepth = rootdepth 
        self.digest = self.md5digest( nodepath[0:leafdepth] )
        self.parent_digest = self.md5digest( nodepath[0:rootdepth] )

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
        return "\n".join(lines)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    path = os.path.expandvars("$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae")
    import collada 
    dae = collada.Collada(path)

    log.info("dae parse completed, now create VNode heirarchy ")
    if os.path.exists(VNode.pkpath):
        VNode.load()
    else:
        top = dae.scene.nodes[0]
        VNode.recurse(top)
        VNode.save()

    VNode.walk()



