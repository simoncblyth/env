#!/usr/bin/env python
"""
Geant4 level interpretation of G4DAEWrite exported pycollada geometry
=======================================================================

Attempt to create the VNode heirarcy out of a raw collada traverse 
without monkeying around.


Web server access
------------------

Access the geometry via a web server (using webpy) from curl or browser
avoiding the overhead of parsing/traversing the entire .dae just 
to look at a selection of volumes::

     http://localhost:8080/dump/1000:1100
     http://localhost:8080/dump/0:10,100:110?ancestors=1

From CLI remember to escape the ampersand::

    curl http://localhost:8080/dump/1000?ancestors=1\&other=yes
    curl http://localhost:8080/dump/__dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..2--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xb35ffb0.1?ancestors=1
    curl http://localhost:8080/dump/__dd__Geometry__AD__lvSST--pvOIL0xb36eb48.1?ancestors=1

TODO:

#. look again at avoiding the ptr references in the DAE ids, 
   as they make references only live until the next .dae export is done

#. use xmlnode elements OR a higher level pycollada approach to piece together .dae 
   sub-selections of the tree of volumes, for visual checking eg with pyglet 


"""
import sys, os, logging, hashlib
import pickle

try:
    import web 
except ImportError:
    web = None

log = logging.getLogger(__name__)

class VNode(object):
    registry = []
    lookup = {}
    idlookup = {}
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

        #if cls.rawcount < 10:
        #    log.info("recurse [%s] %s : %s " % (cls.rawcount, id(node), node ))
        #if not limit is None and cls.rawcount > limit:
        #    log.warn("truncating recurse at rawcount %s " % cls.rawcount )
        #    return

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
    def interpret_ids(cls, arg):
        """
        Interpret an arg like 0:10,400:410,300,40,top.0
        into a list of integer VNode indices 
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
                try:
                    int(arg)
                    ids.append(int(arg))
                except ValueError:
                    node = cls.idlookup.get(arg,None)
                    if node:
                        ids.append(node.index)
                    else:
                        log.warn("failed to lookup VNode for arg %s " % arg)
        return ids


    @classmethod
    def find_uid(cls, bid, decodeNCName=False):
        """
        :param bid: basis ID
        :param decodeNCName: more convenient not to decode for easy URL/cmdline  arg passing without escaping 

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
        cls.idlookup[node.id] = node   
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
        if andself:
            yield self
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

    def matdict(self):
        if not hasattr(self, 'boundgeom'):
            return {}
        bg = self.boundgeom
        msi = bg.materialnodebysymbol.items()
        assert len(msi) == 1 
        symbol, matnode= msi[0]
        matid = matnode.target.id
        return dict(matid=matid, symbol=symbol)

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
        lines.append("VNode(%s,%s)[%s,%s] %s " % (self.rootdepth,self.leafdepth,self.index, self.id, matdict.get('matid',"-") ) )
        lines.extend(self.primitives())
        return "\n".join(lines)

    __repr__ = __str__


def parse_collada( path , usecache=False ):
    """
    :param path: to collada file

    #. parse the .dae with pycollada and obtain list of bound geometry
    #. traverse the raw pycollada node tree, creating an easier to navigate VNode heirarchy 
       which has one VNode per bound geometry  
    #. cross reference between the bound geometry list and the VNode tree

    """
    import collada 
    path = os.path.expandvars(path)
    log.info("pycollada parse %s " % path )
    dae = collada.Collada(path)
    log.info("pycollada parse completed ")
    boundgeom = list(dae.scene.objects('geometry'))
    top = dae.scene.nodes[0]
    log.info("pycollada binding completed, found %s  " % len(boundgeom))

    log.info("create VNode heirarchy ")
    if usecache and os.path.exists(VNode.pkpath):
        VNode.load()
    else:
        VNode.recurse(top, limit=None)
        VNode.summary()
        if usecache:
            VNode.save()
    
    VNode.indexlink( boundgeom )
    #VNode.walk()
    return boundgeom


class _index:
    def GET(self):
        return "_index %s " % len(VNode.registry)

class _dump:
    def GET(self, arg):
        ids = VNode.interpret_ids(arg)
        req = web.input()
        hdr = ["_dump [%s] => [%s] ids " % (arg, len(ids)), "req %s " % req , "" ]
        vnode_ = lambda _:VNode.registry[_] 

        out = []
        if not hasattr(req,'ancestors'):
            out = map(vnode_, ids)
        else:
            amode = req.ancestors
            log.info("amode %s " % amode )
            for id in ids:
                node = vnode_(id)
                out.append(id)
                out.append(node)
                for _ in node.ancestors():
                    out.append(_) 

        return "\n".join(map(str,hdr+out))

class Defaults(object):
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    loglevel = "INFO"
    logpath = None
    daepath = "$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae"
    webserver = False


def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=defopts.logpath )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-f", "--logformat", default=defopts.logformat )
    op.add_option("-p", "--daepath", default=defopts.daepath )
    op.add_option("-w", "--webserver", action="store_true", default=defopts.webserver )

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
        try: 
            logging.basicConfig(format=opts.logformat,level=level)
        except TypeError:
            hdlr = logging.StreamHandler()              # py2.3 has unusable basicConfig that takes no arguments
            formatter = logging.Formatter(opts.logformat)
            hdlr.setFormatter(formatter)
            log.addHandler(hdlr)
            log.setLevel(level)
        pass
    pass
    log.info(" ".join(sys.argv))
    daepath = os.path.expandvars(os.path.expanduser(opts.daepath))
    if not daepath[0] == '/':
        opts.daepath = os.path.join(os.path.dirname(__file__),daepath)
    else:
        opts.daepath = daepath 
    assert os.path.exists(daepath), (daepath,"DAE file not at the new expected location, please create the directory and move the .dae  there, please")
    return opts, args


def webserver():
    log.info("starting webserver ")
    urls = ( 
             '/', '_index', 
             '/dump/(.+)?', '_dump' )
    for i in range(len(urls)/2):
        log.info("%-30s %s " % (urls[i*2+0], urls[i*2+1])) 
    pass    
    app = web.application(urls, globals())
    app.run() 

def main():
    opts, args = parse_args(__doc__) 
    log.info("reading %s " % opts.daepath )
    boundgeom = parse_collada( opts.daepath )
    if opts.webserver:
        webserver()

if __name__ == '__main__':
    main()


