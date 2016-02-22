#!/usr/bin/env python
import logging, hashlib, sys, os
import numpy as np
np.set_printoptions(precision=2) 

log = logging.getLogger(__name__)


class Progeny(list):
    pass


TYPCODE = {'Union':10, 'Intersection':20, 'Sphere':3, 'Tubs':4 }


class CSG(object):
    def __init__(self, elem, children=[]):
        """
        :param elem: 
        :param children: comprised of either base elements or other CSG nodes 
        """
        self.ele = elem
        self.children = children
        self.typ = self.ele.__class__.__name__ 

    def progeny_count(self):
        n = 0 
        for c in self.children:
            if type(c) is CSG:
                n += c.progeny_count()
            else:
                n += 1 
            pass
        return n 

    def as_csg(self):
        return [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    @classmethod
    def serialize(cls, data, offset, obj):
        """
        :param data: npy buffer to fill
        :param offset: primary index of buffer at which to start writing 
        :param obj: to serialize
        :return: offset updated to next place to write


        (2,0) typecode

        (3,0) current offset, ie index of the serialized record  
        (3,1) number of children
        (3,2) index corresponding to first child
        (3,3) index corresponding to last child (including recursion)   

        """
        pass
        nchild = 0 
        if type(obj) is CSG:
            log.info("**serialize offset %s typ %s [%s] " % (offset,obj.typ,repr(obj)))
            nchild = len(obj.children)
            payload = obj.ele if nchild == 0 else obj    # CSG nodes wrapping single elem, kinda different 
        else:
            payload = obj
        pass
        base = offset

        data[base] = payload.as_csg()
        data[base].view(np.int32)[2,0] = TYPCODE.get(payload.typ, -1 )
        data[base].view(np.int32)[3,0] = base
        offset += 1 

        if nchild > 0:
            data[base].view(np.int32)[3,1] = nchild
            data[base].view(np.int32)[3,2] = base + 1
            for c in obj.children:
                offset = cls.serialize(data, offset, c)
            pass
            data[base].view(np.int32)[3,3] = offset - 1    # after the recursion
        pass
        return offset 


    def progeny(self):
        pgs = Progeny()
        nchild = len(self.children)

        if nchild == 0:
            pgs.extend([self.ele])
        else:
            for c in self.children:
                if type(c) is CSG:
                    pg = c.progeny()
                    pgs.extend(pg)
                else:
                    pgs.extend([c])
                pass
            pass
        pass
        return pgs



    def __repr__(self):
        pc = self.progeny_count()
        return "CSG node, %s children, %s progeny [%s] " % ( len(self.children), pc, repr(self.ele)) + "\n" + "\n".join(map(lambda _:"    " + repr(_), self.children))



 
