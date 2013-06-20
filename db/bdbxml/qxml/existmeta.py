#!/usr/bin/env python
"""
NB this is bare dbxml without the boilerplate factored away 

http://www.oracle.com/technetwork/database/berkeleydb/xml-faq-088319.html#HowcanIqueryadocumentwithoutputtingitinthedatabase

"""
import os, logging
log = logging.getLogger(__name__)

try:
    import dbxml
except ImportError:
    dbxml = None


class ExistMeta(object):
    metaname = '__contents__.xml'
    namespace = 'http://exist.sourceforge.net/NS/exist'
    def __init__(self, mgr ):
        qctx = mgr.createQueryContext()
        qctx.setNamespace("", self.namespace) # empty prefix sets uri as default namespace
        qe = mgr.prepare("//resource", qctx )
	pass
	self.mgr = mgr
        self.qe = qe
	self.qctx = qctx
    def __call__(self, path):
	"""
        Query against xml file without placing in dbxml container
	"""
        assert path.endswith(self.metaname), "expecting path to exist metadata file not:%s " % path 
        mgr = self.mgr
	qe = self.qe
        qctx = self.qctx
	pass
        doc = mgr.createDocument()
        doc.setContent( open(path).read() )
        ctx = dbxml.XmlValue(doc)
        meta = {}
        for v in qe.execute( ctx, qctx ):
	    d = dict([(att.getNodeName(),att.getNodeValue()) for att in v.getAttributes()])
	    meta[d['name']] = d
        return meta

        

def existmeta( mgr , path ):
    """
    TODO: avoid expensive query preparation at every call
    """
    qctx = mgr.createQueryContext()
    qctx.setNamespace("","http://exist.sourceforge.net/NS/exist" ) # empty prefix sets uri as default namespace
    qe = mgr.prepare("//resource", qctx )

    # query against xml file without placing in dbxml container
    doc = mgr.createDocument()
    doc.setContent( open(path).read() )
    ctx = XmlValue(doc)

    meta = {}
    for v in qe.execute( ctx, qctx ):
	d = dict([(att.getNodeName(),att.getNodeValue()) for att in v.getAttributes()])
	meta[d['name']] = d
    return meta



if __name__ == '__main__':

    path = "/data/heprez/data/backup/part/localhost/last/db/hfagc_system/__contents__.xml"
    try:
        mgr = dbxml.XmlManager()
        #meta = existmeta( mgr , path )

        em = ExistMeta(mgr)
	meta = em(path)

	print meta

    except dbxml.XmlException, e:
	print "XmlException (", e.exceptionCode,"): ", e.what
	if e.exceptionCode == DATABASE_ERROR:
	    print "Database error code:",e.dbError
    pass 

    	


