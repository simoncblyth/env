#!/usr/bin/env python
"""
Usage for full ingests::

	./exist2qxml.py

For selective ingests, eg into container with tag 'sys'::

         EXIST2QXML_SELECT=sys@@http://localhost/servlet/db/hfagc_system/v2qtags.xml ./exist2qxml.py
         EXIST2QXML_SELECT=sys@@http://localhost/servlet/db/hfagc_system/qtag2latex.xml ./exist2qxml.py

Configured by the file pointed to by QXML_CONFIG in particular the below:: 

	[container.source]
	source = /tmp/check/db/hfagc_prod/end_of_2011/indv

	[container.path]
	path = /tmp/hfagc/avg.dbxml

	[container.tag]
	tag = avg


To suppress the leading slash in db names, supply a trailing slash in the source.
This is useful for the sys container as it contains no sub-collections and not
having slashes in names affords shortcut doc access.

All non-metadata xml files beneath the `srcdir` are ingested into container dbxml 
at the path specified which is subsequently referred to with qxml via the 
configured tag or alias eg with::

   collection('avg')/dbxml:metadata('dbxml:name')


TODO
=====

 1) split ingest_dir into
   
       `ingest_backup` 
             with source the path to the __contents__.xml 
	     operating via traversing __contents__.xml files 

       `ingest_dir` 
             loading generic directories of xml files via directory walk

  2) provide argument handling to propagate single files across

exist2qxml.py



"""
import os, logging
log = logging.getLogger(__name__)
from dbxml import *
from config import qxml_config

# os.path.relpath only from py26
relpath = lambda path,root:path[len(root):]      # keep leading slash to allow referring to '/' as root of all
	
from existmeta import ExistMeta
from common import existsDoc, urlDoc, ExistDirQuery


class ExistWalk(object):
    """
    Provides remote pulling of exist resources into local dbxml containers using exist servlet urls.
    This avoids the traditional propagation approach of going via an exist backup::

         cd ~/heprez/backup/part
         ant -Dbackup.dir=/tmp/check part-backup-hfagc-prod-once-only

    """
    def __init__(self, mgr ): 
        self.mgr = mgr
        self.edq = ExistDirQuery(mgr)

    def walk( self, dirurl ):
	"""
	:param dirurl: exist servlet directory URL, typically ending in a slash  

        recursive walk of exist servlet urls such as http://localhost/servlet/db/hfagc/
        following os.walk pattern
	"""
        collections, resources = self.edq( dirurl )
        yield (dirurl, collections, resources )      # topdown traverse
        for collection in collections:
	    for x in self.walk( "%s%s/" % ( dirurl, collection )):
                yield x    




def ingest_url( tag, srcurl, dbxml , srcpfx=None ):
    """	
    :parm tag: alias string of the container to be created
    :param srcurl: exist servlet url such as  http://localhost/servlet/db/hfagc/
    :param dbxml: path of dbxml container to be created
    :param srcpfx:  src url prefix allowing restriction of urls to be ingested 
                    eg provide full resource url to ingest just that into the container

    ISSUE: metadata attribute naming mismatch 

    Attribute names from servlet dir listings::

	<exist:result xmlns:exist="http://exist.sourceforge.net/NS/exist">
  	    <exist:collection name="/db/hfagc/lhcb/yasmine" owner="yasmine" group="lhcb" permissions="rwur-ur-u">
	        <exist:resource name="lhcb_winter2011_BsDst1Xmunu.xml" created="Apr 7, 2012 04:03:00" last-modified="Apr 8, 2012 01:54:31" owner="yasmine" group="lhcb" permissions="rwur-ur--"/>
	        <exist:resource name="lhcb_winter2011_Lb2Lcpipipi.xml" created="Apr 7, 2012 05:26:59" last-modified="Apr 7, 2012 06:10:05" owner="yasmine" group="lhcb" permissions="rwur-ur--"/>
                ...

    From backup __contents__.xml::

        <collection xmlns="http://exist.sourceforge.net/NS/exist" name="/db/hfagc/lhcb/yasmine" owner="yasmine" group="lhcb" mode="755" created="2012-03-19T18:57:26+08:00">
                <resource type="XMLResource" name="lhcb_winter2011_B02DK.xml" owner="yasmine" group="lhcb" mode="754" created="2012-04-04T14:15:30+08:00" modified="2012-04-08T01:55:23+08:00" filename="lhcb_winter2011_B02DK.xml" mimetype="text/xml"/>
                <resource type="XMLResource" name="lhcb_winter2011_B02Dupipipi.xml" owner="yasmine" group="lhcb" mode="754" created="2012-04-07T16:47:43+08:00" modified="2012-04-07T16:54:29+08:00" filename="lhcb_winter2011_B02Dupipipi.xml" mimetype="text/xml"/>


    """
    if not(srcurl.startswith("http://")):
	log.debug("skipping tag %s dbxml %s as invalid srcurl %s " % ( tag, dbxml, srcurl ))
	return
    else:
        log.info("ingest %s creating %s from xml files from %s " % ( tag, dbxml, srcurl ))
	pass

    try:
        mgr = XmlManager()
        ing = ExistWalk(mgr)
        update = srcpfx != None
        if update:
	    cont = mgr.openContainer(dbxml)
        else:  
	    cont = mgr.createContainer(dbxml)
	uctx = mgr.createUpdateContext()
 
        for (urlpath, collections, resources) in ing.walk( srcurl ):
	    rurl = relpath(urlpath, srcurl )   
	    for d in resources:
		name = d['name']    
                p = os.path.join(urlpath,name)  
                n = os.path.join(rurl,name)
		if srcpfx: 
		    if p.startswith(srcpfx):    # restrict ingest via srcpfx
			log.info("selective ingest of %s %s " % (p,n) )    
		    else:
		        continue
                else:
                    log.info("ingesting %s %s " % ( p, n ))

		doc = urlDoc( mgr, p, name=n, meta=d )
		if existsDoc( n, cont):
		    log.info("deleting pre-existing document %s " % n )	
		    cont.deleteDocument( n , uctx ) 	
                cont.putDocument( doc , uctx, 0 ) 

    except XmlException, e:
	print "XmlException (", e.exceptionCode,"): ", e.what
	if e.exceptionCode == DATABASE_ERROR:
	    print "Database error code:",e.dbError
    pass 



def ingest_dir( tag, srcdir , dbxml , srcpfx='/' ):
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
		if not(p.startswith(srcpfx)):    # restrict ingest via srcpfx
		    continue	
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
    """
    resort to EXIST2QXML_SELECT environ as specific to exist2qxml 
    and do not want to mess with qxml_config arg parsing for this 	
    """
    cfg = qxml_config()
    tagsrc = cfg['source'].keys()
    tagcon = cfg['containers'].keys()
    assert tagsrc == tagcon , (tagsrc, tagcon )

    select = os.environ.get('EXIST2QXML_SELECT',None )	   

    srcpfx = {}
    if select:
	stag, pfx = select.split("@@")    
        log.info("select %s stag %s pfx %s " % ( select, stag, pfx ))
        srcpfx[stag] = pfx

    for tag in tagsrc:
	src = cfg['source'][tag]    
	dbxml  = cfg['containers'][tag]    
        if os.path.exists(dbxml) and tag not in srcpfx:
            log.warn("tag %s dbxml \"%s\" exists already : delete it and rerun to update from src \"%s\"  " % ( tag, dbxml, src ))     
            continue
	if src.startswith('http://'):  
	    srcpfx_ = srcpfx.get(tag, None) 
	    log.info("using srcpfx_ %s " % srcpfx_ )
	    ingest_url( tag, src, dbxml , srcpfx_ )	   
 	else:	
            ingest_dir( tag, src, dbxml )	   
	pass    
    pass

if __name__ == '__main__':
    pass	
    main()
     
