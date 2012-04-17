#!/usr/bin/env python
from dbxml import *

def existsDoc( docname, cont ):
    try:
        doc = cont.getDocument(docname, DBXML_LAZY_DOCS)
        ret = True
    except XmlException, e:
    	if e.exceptionCode == DOCUMENT_NOT_FOUND:
            ret = False
        else:
    	    throw	
    return ret


if __name__ == '__main__':

    from config import qxml_config
    cfg = qxml_config()
    mgr = XmlManager()
    cnt = mgr.openContainer(cfg['containers']['sys'])

    checks = 'v1qtags.xml v2qtags.xml v3qtags.xml'
    for name in checks.split():
        exists = existsDoc( name , cnt )
	print exists, name


