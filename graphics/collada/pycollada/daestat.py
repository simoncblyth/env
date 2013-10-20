#!/usr/bin/env python
"""
DAESTAT
========

Basic stats for a Collada .dae file, number of nodes etc..

"""
import lxml.etree as ET
import sys, os, logging
tostring_ = lambda _:ET.tostring(_)
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
COLLADA_NS='http://www.collada.org/2005/11/COLLADASchema'
tag_ = lambda _:"{%s}%s" % ( COLLADA_NS, _ )
xpath_ = lambda _:"./"+"/".join(map(tag_,_.split("/"))) 
id_ = lambda _:_.attrib.get('id',None)

log = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv)>1:
        path = sys.argv[1]
    else:    
        path = '$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae'

    log.info("parsing %s " % path)    
    root = parse_(path)
    created = root.find(xpath_("asset/created")).text
    print "created %s " % created

    fmt = "%-70s %-10s %-10s " 
    print fmt % ( "top element", "#ids", "#uids" )
    for topd in root.findall("./*"):
        subs = topd.findall("./*")
        ids = map(id_, subs )
        uids = set(ids)
        print fmt  % ( topd.tag, len(ids), len(uids) )

if 0:    
    for lib in "library_effects/effect library_materials/material library_geometries/geometry".split():
        print
        print lib 
        for i, id in enumerate(map(id_, root.findall(xpath_(lib)))):
            print "%5s %s " % ( i, id )





    


