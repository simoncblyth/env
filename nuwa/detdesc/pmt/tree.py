#!/usr/bin/env python
import logging, hashlib
import numpy as np
from dd import Dddb 

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
            self._parts = self.lv.parts()
        return self._parts

    def num_parts(self):
        parts = self.parts()
        return len(parts)

    def copy_parts(self, data, offset):
        """
        # use 4th slots of bbox min/max for integer codes
        """
        for i,part in enumerate(self.parts()):
            data[offset+i] = part.as_quads()
            data[offset+i].view(np.int32)[2,3] = part.typecode 
            data[offset+i].view(np.int32)[3,3] = self.index   
 

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
    def save_parts(cls, path, limit=None):
        tnodes = cls.num_nodes() 
        tparts = cls.num_parts() 
        log.info("tnodes %s tparts %s " % (tnodes, tparts))

        data = np.zeros([tparts,4,4],dtype=np.float32)
        offset = 0 
        for i in range(tnodes):
            node = tree.get(i)
            node.copy_parts(data, offset)    
            nparts = node.num_parts() 
            log.info("i %s %s %s " % (i, nparts, repr(node))) 
            offset += nparts
        pass


        if limit is not None:
            log.warning("save_parts limited to %d parts " % limit )
            data = data[:limit]

        rdata = data.reshape(-1,4) 
        log.info("save_parts to %s reshaped from %s to %s for easier GBuffer::load  " % (path, repr(data.shape), repr(rdata.shape)))



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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    g = Dddb.parse("$PMT_DIR/hemi-pmt.xml")
    tree = Tree(g.logvol_("lvPmtHemi")) 
    tree.save_parts("/tmp/hemi-pmt-parts.npy", 3)
    #tree.save_parts("/tmp/hemi-pmt-parts.npy")




