#!/usr/bin/env python
from dbxml import *
from existmeta import ExistMeta
import os
from datetime import datetime

def existsDoc( docname, cont ):
    """
    :param docname:
    :param cont: 
    """
    try:
        doc = cont.getDocument(docname, DBXML_LAZY_DOCS)
        ret = True
    except XmlException, e:
        if e.exceptionCode == DOCUMENT_NOT_FOUND:
            ret = False
        else:
            throw    
    return ret

def urlDoc(mgr, url, other="", name=None, meta=None ):
    """
    :param mgr: XmlManager instance
    :param url: to be loaded
    """
    if url.startswith('http://'):  
        stm = mgr.createURLInputStream(other, url)
    else:
        stm = mgr.createLocalFileInputStream(url)

    doc = mgr.createDocument()
    doc.setContentAsXmlInputStream( stm )
    if not name:
        name = os.path.basename(url) 
    doc.setName( name )
    if meta:
        for key, val in meta.items():
            doc.setMetaData( ExistMeta.namespace , key, XmlValue(val) )
    return doc


class DateTimeFix(object):
    def __init__(self, ifmt="%b %d, %Y %H:%M:%S" , ofmt="%Y-%m-%dT%H:%M:%SZ" ):
        """
        :param ifmt: default is format used in exist:modified
        :param ofmt: default is format that works with xs:dateTime
        """
        self.ifmt = ifmt
        self.ofmt = ofmt 
    def __call__(self, s ):
        t = datetime.strptime(s, self.ifmt)
        return t.strftime(self.ofmt)   


class ExistDirQuery(object):
    def __init__(self, mgr ):     
        qctx = mgr.createQueryContext()
        qctx.setNamespace("", ExistMeta.namespace )      # empty prefix sets uri as default namespace

        self.query = mgr.prepare("/result/collection/*", qctx )
        self.qctx = qctx
        self.mgr = mgr
        self.dtf = DateTimeFix()

    def __call__(self, dirurl ):
        """
        Reads exist servelet directory url and performs xquery 
        on the returned xml

        :param mgr:
        :param dirurl: 

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
        if not(dirurl.endswith('/')):
            dirurl = dirurl + '/'
        doc = urlDoc( self.mgr, dirurl )
        ctx = XmlValue(doc)

        collections = []
        resources = []
        res = self.query.execute( ctx, self.qctx )
        for v in res: 
            d = dict([(att.getNodeName(),att.getNodeValue()) for att in v.getAttributes()])

            ## fix to match attribute name with those from backups
            if 'modified' not in d and 'last-modified' in d:
                #d['modified'] = d['last-modified']
                d['modified'] = self.dtf(d['last-modified'])
                del d['last-modified']

            name = v.getNodeName()
            if name == 'exist:resource':   
                resources.append( d )    
            elif name == 'exist:collection':         
                collections.append( d['name'] )    
            else:
                log.warn("ignoring unhandled element %s " % name )     
            pass
        return collections, resources


if __name__ == '__main__':

    from config import qxml_config
    cfg = qxml_config()
    mgr = XmlManager()
    cnt = mgr.openContainer(cfg['containers']['sys'])

    checks = 'v1qtags.xml v2qtags.xml v3qtags.xml'
    for name in checks.split():
        exists = existsDoc( name , cnt )
        print exists, name


