#!/usr/bin/env python
import logging, hashlib, sys, os
import numpy as np
np.set_printoptions(precision=2) 
from dd import Dddb, Part

log = logging.getLogger(__name__)

class Node(object):
    @classmethod
    def md5digest(cls, volpath ):
        """  
        Use of id means that will change from run to run. 
        """
        dig = ",".join(map(lambda _:str(id(_)),volpath))
        dig = hashlib.md5(dig).hexdigest() 
        return dig

    @classmethod
    def create(cls, volpath ):
        assert len(volpath) >= 2 
        node = cls(volpath) 

        ndig = node.digest   ; assert ndig not in Tree.registry 
        node.index  = len(Tree.registry)

        Tree.byindex[node.index] = node 
        Tree.registry[ndig] = node

        node.parent = Tree.lookup(node.pdigest)
        if node.parent:
            node.parent.add_child(node)  

        node.pv = volpath[-2] if type(volpath[-2]).__name__ == "Physvol" else None  # tis None for root
        node.lv = volpath[-1] if type(volpath[-1]).__name__ == "Logvol" else None
        assert node.lv

        node.posXYZ = node.pv.find_("./posXYZ") if node.pv is not None else None

        #node.dump("visitWrap_")
        return node

    def __init__(self, volpath):
        # set by Tree
        self.parent = None
        self.index = None
        self.posXYZ = None
        self.children = []
        self.lv = None
        self.pv = None
        self._parts = None

        self.volpath = volpath
        self.digest = self.md5digest( volpath[0:len(volpath)] )
        self.pdigest = self.md5digest( volpath[0:len(volpath)-2] )

    def visit(self, depth):
        log.info("visit depth %s %s " % (depth, repr(self)))

    def traverse(self, depth=0):
        self.visit(depth)
        for child in self.children:
            child.traverse(depth+1)

    def parts(self):
        """
        Divvy up geometry into parts that 
        split "intersection" into union lists. This boils
        down to judicious choice of bounding box according 
        to intersects of the source gemetry.
        """
        if self._parts is None:
            _parts = self.lv.parts()
            for p in _parts:
                p.node = self
            pass
            self._parts = _parts 
        pass
        return self._parts

    def num_parts(self):
        parts = self.parts()
        return len(parts)



    def add_child(self, child):
        log.debug("add_child %s " % repr(child))
        self.children.append(child)

    def dump(self, msg="Node.dump"):
        log.info(msg + " " + repr(self))
        #print "\n".join(map(str, self.geometry))   

    def __repr__(self):
        return "Node %2d : dig %s pig %s : %s : %s " % (self.index, self.digest[:4], self.pdigest[:4], repr(self.volpath[-1]), repr(self.posXYZ) ) 



class Tree(object):
    """
    Following pattern of assimpwrap-/AssimpTree 
    transforming tree from  pv/lv/pv/lv/.. to   (pv,lv)/(pv,lv)/ ...

    Note that the point of this is to create a tree at the 
    desired granularity (with nodes encompassing PV and LV)
    which can be serialized into primitives for analytic geometry ray tracing.
    """
    registry = {}
    byindex = {}

    @classmethod
    def lookup(cls, digest):
        return cls.registry.get(digest, None)  

    @classmethod
    def get(cls, index):
        return cls.byindex.get(index, None)  

    @classmethod
    def num_nodes(cls):
        assert len(cls.registry) == len(cls.byindex)
        return len(cls.registry)

    @classmethod
    def num_parts(cls):
        nn = cls.num_nodes()
        tot = 0 
        for i in range(nn):
            node = cls.get(i)
            tot += node.num_parts()
        pass
        return tot

    @classmethod
    def save_parts(cls, path, explode=0., container=False, container_factor=3., reshape=False):
        pdir = os.path.dirname(path)
        if not os.path.exists(pdir):
            os.makedirs(pdir)

        tnodes = cls.num_nodes() 
        tparts = cls.num_parts() 
        log.info("tnodes %s tparts %s " % (tnodes, tparts))

        # collect Part instances from each of the nodes into list
        parts = []
        for i in range(tnodes):
            node = tree.get(i)
            parts.extend(node.parts())    
        pass
        assert len(parts) == tparts          

        if container:
            container = Part.make_container(parts, factor=container_factor)
            parts.extend([container])
            # match order of what GMergedeMesh::combine does with the GTestBox
        pass

        # serialize parts into array, converting relationships into indices
        data = np.zeros([len(parts),4,4],dtype=np.float32)
                 
        indices = map(lambda p:p.node.index, filter(lambda p:p.node is not None, parts))
        fabricated_nodeindex = max(indices) + 1  # rustle up next valid index, all fabricated parts will have this index
        log.info("indices: %s fabricated_nodeindex: %s " % (repr(indices), fabricated_nodeindex))

        for i,part in enumerate(parts):
            if part.node is None:       # eg for the fabricated container box
                nodeindex = fabricated_nodeindex
            else:
                nodeindex = part.node.index
            pass
            index = i + 1   # 1-based index, where parent 0 means None
            if part.parent is not None:
                parent = parts.index(part.parent) + 1   # lookup index of parent in parts list  
            else:
                parent = 0 
            pass
            data[i] = part.as_quads()

            if explode>0:
                dx = i*explode
 
                data[i][0,0] += dx
                data[i][2,0] += dx
                data[i][3,0] += dx

            data[i].view(np.int32)[1,1] = index  
            data[i].view(np.int32)[1,2] = parent
            data[i].view(np.int32)[1,3] = part.flags    # used in intersect_ztubs
            # use the w slot of bb min, max for typecode and solid index
            data[i].view(np.int32)[2,3] = part.typecode 
            data[i].view(np.int32)[3,3] = nodeindex   
        pass

        if reshape:
            rdata = data.reshape(-1,4) 
            log.debug("save_parts to %s reshaped from %s to %s for easier GBuffer::load  " % (path, repr(data.shape), repr(rdata.shape)))
        else:
            rdata = data 

        log.info("saving to %s shape %s " % (path, repr(rdata.shape)))
        np.save(path, rdata) 

    def traverse(self):
        self.wrap.traverse()

    def __init__(self, base):
        self.base = base
        self.wrap = None
        ancestors = [self]   # dummy top "PV", to regularize striping: TOP-LV-PV-LV 
        self.wrap = self.traverseWrap_(self.base, ancestors)

    def traverseWrap_(self, vol, ancestors):
        """
        #. vital to make a copy with [:] as need separate volpath for every node
        #. only form wrapped nodes at Logvol points in the tree
           in order to have regular TOP-LV-PV-LV ancestry, 
           but traverse over all nodes of the source tree
        """
        volpath = ancestors[:] 
        volpath.append(vol) 

        ret = None
        if type(volpath[-1]).__name__ == "Logvol":
            ret = self.visitWrap_(volpath)

        for child in vol.children():
            self.traverseWrap_(child, volpath)
        pass 
        return ret

    def visitWrap_(self, volpath):
        log.debug("visitWrap_ %s : %s " % (len(volpath), repr(volpath[-1])))
        return Node.create(volpath)


usage = """
Argument handling

           # default is all parts  
    0:3    # just first 3 spheres of solid 0 
    0:4    # 3 sph and tubs of solid 0  

"""

if __name__ == '__main__':
    format_ = "[%(filename)s +%(lineno)3s %(funcName)20s ] %(message)s" 
    logging.basicConfig(level=logging.INFO, format=format_)


    g = Dddb.parse("$PMT_DIR/hemi-pmt.xml")
    tree = Tree(g.logvol_("lvPmtHemi"))

    dest = os.path.expandvars("$IDPATH/GPmt/0/GPmt.npy");
    #dest = "/tmp/hemi-pmt-parts.npy"

    tree.save_parts(dest, explode=0., container=True, container_factor=3.) 




