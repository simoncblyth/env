#!/usr/bin/env python
"""


"""
import os, sys
import lxml.etree as ET
import numpy as np
np.set_printoptions(threshold=5)   # eliding mid elements of large arrays 

parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"


if __name__ == '__main__':

    args = sys.argv[1:]
    narg = len(args)

    pth, qid, qpr = "g4_00.dae","", ""
    if narg > 0:pth = args[0]
    if narg > 1:qid = args[1]
    if narg > 2:qpr = args[2]

    xml = parse_(pth)
    for mat in  xml.findall(".//{%s}material" % COLLADA_NS ):
        extra = mat.find(".//{%s}extra" % COLLADA_NS ) 

        if extra is not None:
            id = mat.attrib['id']

            if len(qid) == 0 or id.startswith(qid):
                props = extra.findall(".//{%s}property" % COLLADA_NS )
                data = {} 
                for matrix in extra.findall(".//{%s}matrix" % COLLADA_NS ):
                    ref = matrix.attrib['name']
                    assert matrix.attrib['coldim'] == '2'
                    data[ref] = matrix.text

                for prop in props:
                    pr = prop.attrib['name']
                    if len(qpr) == 0 or pr.startswith(qpr):
                        ref = prop.attrib['ref']
                        s = data[ref]
                        if s == None:
                            print "failed to deref %s " % ref
                            b = None
                        else:
                            a = np.fromstring(s, dtype=float, sep=' ')
                            assert len(a) % 2 == 0
                            b = a.reshape((-1,2))
                        pass
                        print "    %-30s : (%s) " % ("%s.%s" % (id,pr), len(b) )
                        print b





