#!/usr/bin/env python
"""
"""
import pprint
from qxml import QXML

if __name__ == '__main__':
    qx = QXML()
    q = "collection('dbxml:/sys')/*[dbxml:metadata('dbxml:name')='pdgs.xml' or dbxml:metadata('dbxml:name')='extras.xml' ]//glyph"
    glyph = {}
    for v in qx(q):
	d = dict([(att.getNodeName(),att.getNodeValue()) for att in v.getAttributes()])    
        glyph[d['code']] = d['latex']
        pass
    print pprint.pformat(glyph)




