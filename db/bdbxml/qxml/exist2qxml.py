#!/usr/bin/env python
"""
Configured by the file pointed to by QXML_CONFIG in particular the below:: 

	[container.srcdir]
	srcdir = /tmp/check/db/hfagc_prod/end_of_2011/indv

	[container.path]
	path = /tmp/hfagc/avg.dbxml

	[container.tag]
	tag = avg


All non-metadata xml files beneath the `srcdir` are ingested into container dbxml 
at the path specified which is subsequently referred to with qxml via the 
configured tag or alias eg with::

   collection('avg')/dbxml:metadata('dbxml:name')


TODO: split ingest_dir into
   
       `ingest_backup` 
             with source the path to the __contents__.xml 
	     operating via traversing __contents__.xml files 

       `ingest_dir` 
             loading generic directories of xml files via directory walk

"""
import os, logging
log = logging.getLogger(__name__)
from dbxml import *
from config import qxml_config

# os.path.relpath only from py26
relpath = lambda path,root:path[len(root):]      # keep leading slash to allow referring to '/' as root of all
	
from existmeta import ExistMeta

class ExistIngester(object):
    def __init__(self, mgr ): 
        self.mgr = mgr
        qctx = mgr.createQueryContext()
        qctx.setNamespace("", ExistMeta.namespace )      # empty prefix sets uri as default namespace
        query = mgr.prepare("/result/collection/*", qctx )
        self.query = query
        self.qctx = qctx


    def urldoc(self, url, other=""):
	"""
	"""
        stm = self.mgr.createURLInputStream(other, url)
        doc = self.mgr.createDocument()
        doc.setContentAsXmlInputStream( stm )
        return doc


    def walk( self, dirurl ):
	"""
	:param dirurl: exist servlet directory URL, ending in slash  

        recursive walk of exist servlet urls such as http://localhost/servlet/db/hfagc/
        following os.walk pattern

        Each directory url returns XML of structure::

            result
	       collection
	            collection
		    collection
		    resource
		    resource

        Note that the structure differs from the backup __contents__.xml files, 
        although holding the same information  
	
	Why the ns prefixes ?

	"""
	assert dirurl.endswith('/')
	doc = self.urldoc( dirurl )
        ctx = XmlValue(doc)

        collections = []
	resources = []

	for v in self.query.execute( ctx, self.qctx ):
	    d = dict([(att.getNodeName(),att.getNodeValue()) for att in v.getAttributes()])
            name = v.getNodeName()
	    if name == 'exist:resource':   
		resources.append( d )    
            elif name == 'exist:collection': 		
		collections.append( d['name'] )    
	    else:
		log.warn("ignoring unhandled element %s " % name )     

        yield (dirurl, collections, resources )      # topdown traverse
        for collection in collections:
	    for x in self.walk( "%s%s/" % ( dirurl, collection )):
                yield x    



def ingest_url( tag, srcurl, dbxml ):
    """	
    :parm tag: alias string of the container to be created
    :param srcurl: exist servlet url such as  http://localhost/servlet/db/hfagc/
    :param dbxml: path of dbxml container to be created
    """
    if not(srcurl.startswith("http://")):
	log.debug("skipping tag %s dbxml %s as invalid srcurl %s " % ( tag, dbxml, srcurl ))
	return
    else:
        log.info("ingest %s creating %s from xml files from %s " % ( tag, dbxml, srcurl ))
	pass

    try:
        mgr = XmlManager()
        ing = ExistIngester(mgr)
	cont = mgr.createContainer(dbxml)
	ctx = mgr.createUpdateContext()
 
        for (urlpath, collections, resources) in ing.walk( srcurl ):
	    rurl = '/' + relpath(urlpath, srcurl )   
            print rurl
	    for d in resources:
		name = d['name']    
                p = os.path.join(urlpath,name)  
                n = os.path.join(rurl,name)
	        print p, n	
		doc = ing.urldoc( p )
		doc.setName(n)
		for key, val in d.items():
                    doc.setMetaData( ExistMeta.namespace , key, XmlValue(val) )
		    pass
                cont.putDocument( doc , ctx, 0 ) 

    except XmlException, e:
	print "XmlException (", e.exceptionCode,"): ", e.what
	if e.exceptionCode == DATABASE_ERROR:
	    print "Database error code:",e.dbError
    pass 



def ingest_dir( tag, srcdir , dbxml ):
    """
    :parm tag: alias string of the container to be created
    :param srcdir: exist backup directory to ingest into dbxml container
    :param dbxml: path of dbxml container to be created

    """
    if srcdir == "":
	log.debug("skipping tag %s dbxml %s as invalid srcdir " % ( tag, dbxml ))
	return
    elif not(os.path.isdir(srcdir)):
	log.warn("srcdir \"%s\" does not exist skip ingest into %s " % ( srcdir , dbxml ))     
	return
    else:
        log.info("ingest %s creating %s from xml files from %s " % ( tag, dbxml, srcdir ))
	pass

    try:
        mgr = XmlManager()
	xmeta = ExistMeta(mgr)
        metaname = ExistMeta.metaname
	cont = mgr.createContainer(dbxml)
	ctx = mgr.createUpdateContext()
    
        for (dirpath, dirnames, filenames) in os.walk( srcdir ):
	    rdir = relpath(dirpath, srcdir )

            if metaname in filenames:
		dirmeta = xmeta( os.path.join( dirpath, metaname ) ) 

	    for name in filter(lambda _:_ != metaname, filenames):
                p = os.path.join(dirpath,name)  
                n = os.path.join(rdir,name)
		xm = dirmeta[name]
                print xm
	        stm = mgr.createLocalFileInputStream(p)
                doc = mgr.createDocument()
		doc.setName(n)
		doc.setContentAsXmlInputStream(stm)
		for key, val in xm.items():
                    doc.setMetaData( ExistMeta.namespace , key, XmlValue(val) )
		    pass
                cont.putDocument( doc , ctx, 0 ) 
	    pass


    except XmlException, e:
	print "XmlException (", e.exceptionCode,"): ", e.what
	if e.exceptionCode == DATABASE_ERROR:
	    print "Database error code:",e.dbError
    pass 

def main():
    cfg = qxml_config()
    tagsrc = cfg['source'].keys()
    tagcon = cfg['containers'].keys()
    assert tagsrc == tagcon , (tagsrc, tagcon )
    for tag in tagsrc:
	src = cfg['source'][tag]    
	dbxml  = cfg['containers'][tag]    
        if os.path.exists(dbxml):
            log.warn("tag %s dbxml \"%s\" exists already : delete it and rerun to update from src \"%s\"  " % ( tag, dbxml, src ))     
            continue
	if src.startswith('http://'):  
            ingest_url( tag, src, dbxml )	   
 	else:	
            ingest_dir( tag, src, dbxml )	   
	pass    
    pass

if __name__ == '__main__':
    pass	
    main()
     
