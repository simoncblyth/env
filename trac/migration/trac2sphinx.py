#!/usr/bin/env python
"""
trac2sphinx.py 
================

Canonical usage is from wtracdb.py 

::

    ./trac2sphinx.py $(wtracdb-path) 3D


"""
import logging, sys, re, os, collections, datetime, codecs, copy
log = logging.getLogger(__name__)

from env.trac.migration.resolver import Resolver
from env.trac.migration.tracwiki2rst import TracWiki2RST, ListTagged

class WikiPage(object):
    def __init__(self, db, name):

        tags = map(lambda _:str(_[0]), db("select tag from tags where tagspace=\"wiki\" and name=\"%s\" " % name ))
        rec = db("SELECT version,time,author,text,comment,readonly FROM wiki WHERE name=\"%s\" ORDER BY version DESC LIMIT 1" % name ) 
        version,time,author,text,comment,readonly = rec[0] 

        assert type(author) is unicode
        assert type(text) is unicode
        if comment is not None:
            assert type(comment) is unicode
        pass
        ftime = datetime.datetime.fromtimestamp(time).strftime('%Y-%m-%dT%H:%M:%S' )

        md = collections.OrderedDict()  
        md["name"] = name
        md["version"] = version
        md["time"] = time
        md["ftime"] = ftime
        md["author"] = author
        md["comment"] = comment if comment is not None else ""
        md["tags"] = " ".join(tags)

        self.db = db 

        self.name = name
        self.version = version
        self.time = time
        self.ftime = ftime
        self.author = author
        self.text = text.replace('\r\n','\n')
        self.comment = comment
        self.tags = tags 

        self.metadict = md      

    def complete_ListTagged(self, tgls):
        """
        http://www.sphinx-doc.org/en/stable/markup/inline.html#cross-referencing-documents
        """
        assert type(tgls) is ListTagged, type(tgls)
    
        skips = r"""
        or
        operator=union
        operation=union
        action=union
        """.lstrip().rstrip().split("\n")

        targ = tgls.tags.lstrip().rstrip().replace(","," ")
        tags = filter(lambda _:not _ in skips, targ.split())

        stags = ",".join(map(lambda _:"'%s'" % _, tags ))
        sql = "select distinct name from tags where tagspace=\"wiki\" and tag in (%s) order by name ;" % stags 
        rec = self.db(sql)

        wikitagged = map(lambda _:_[0], rec )
        for nm in wikitagged: 

            psql = "select tag from tags where name = \"%s\" order by tag ;" % nm  
            prec = self.db(psql)
            prec = map(lambda _:_[0], prec )
            ## hmm even when generate taglist only pages, still need to distingish 

            #prst = " ".join(["("] + map(lambda _:":doc:`%s <%s>`" % (_,_), prec ) + [")"])
            #prst2 = " ".join(["("] + map(lambda _:":ref:`%s`" % _, prec ) + [")"])
            #prst3 = " ".join(["("] + map(lambda _:":%s_" % _, prec ) + [")"])

            prst = ""
            prst2 = ""
            prst3 = ""


            tgls.append("%s :doc:`%s` %s %s %s" % (nm,nm, prst, prst2, prst3) )
        pass
        """  
        select distinct tag as t from tags order by tag ;
        select distinct tag as t from tags where t not in ( select distinct name from wiki ) order by tag ;
        select name, count(tag) as n from tags group by tag order by n desc ;
        """

    def __repr__(self):
        return "%5s : %30s : %10s : %15s : %60s : %s " % ( self.version, self.name, self.author, self.time, ",".join(self.tags), self.comment )

    def __unicode__(self):
        return "\n\n".join( [repr(self), unicode(self.text) ] ) 

    def __str__(self):
        return unicode(self).encode('utf-8')



class Sphinx(object):
    def __init__(self, args, db):
        self.args = args 
        
        self.db = db
        self.title = args.title
        self.pages = []

    def add(self, page):
        self.pages.append(page)  

    def write_(self, page):
        """
        http://www.sphinx-doc.org/en/stable/rest.html#source-encoding
        Sphinx assumes source files to be encoded in UTF-8 by default
        """
        rstpath = self.args.resolver.getpath(page.name, ".rst") 
        log.debug("write %s " % rstpath )
        rst = page.rst
        assert type(rst) is unicode
        open(rstpath, "w").write(rst.encode("utf-8"))  
        print "write_ %s " % rstpath

    def write(self):
        for page in self.pages:
            self.write_(page)
        pass
        idx = TracWiki2RST.make_index("index", self.title, self.pages)
        self.write_(idx) 
        log.info("wrote %s pages  %s...%s " % (len(self.pages), self.pages[0].name, self.pages[-1].name))

    def trac2rst_one(self, name):
        log.debug("converting %s " % name )
        txtpath = self.args.resolver.getpath(name, ".txt") 

        wp = WikiPage(self.db, name)
        text_from_db = wp.text
        assert type(text_from_db) is unicode

        text_from_file = codecs.open(txtpath, encoding='utf-8').read() if os.path.exists(txtpath) else None
        assert type(text_from_file) in [unicode, type(None)], (txtpath, type(text_from_file) )

        if text_from_file:
            if text_from_db != text_from_file:
                log.warning("difference between wikitext from db and xmlrpc for %s " % name)
            pass
        pass
        use_text = text_from_file if text_from_file is not None else text_from_db 

        pg = TracWiki2RST.page_from_tracwiki(wp, use_text, self.args)
        self.add(pg)
       

    def trac2rst_all(self):
        names = self.db("select distinct name from wiki") 
        for name, in names:
            self.trac2rst_one(name)
        pass

    def tracdb2rst(self):
        name = self.args.onepage
        if name is None:
            self.trac2rst_all()
        else:
            self.trac2rst_one(name)
        pass




def parse_args(doc):
    import argparse
    parser = argparse.ArgumentParser(doc)

    d = {}
    d['onepage'] = None
    d['rstdir'] = None
    d['tracdir'] = None
    d['title'] = "trac2sphinx.py conversion"
    d['origtmpl'] = None
    d['level'] = "INFO"
    d['dev'] = False
    d['tags'] = None
    d['vanilla'] = False

    parser.add_argument("dbpath", default=None, help="path to trac.db"  ) 
    parser.add_argument("--onepage", default=d['onepage'], help="restrict conversion to single named page for debugging")  
    parser.add_argument("--rstdir", default=d['rstdir'], help="directory to write the converted RST")  
    parser.add_argument("--tracdir", default=d['tracdir'], help="directory named after the repo containing db/trac.db as well as attachements etc.. ")  
    parser.add_argument("--origtmpl", default=d['origtmpl'], help="template of original tracwiki url to provide backlink for debugging, eg http://localhost/tracs/worklow/wiki/%s ")  
    parser.add_argument("--title", default=d['title'] )  
    parser.add_argument("--vanilla", action="store_true", default=d['vanilla'], help="Skip Sphinx extensions to allow plain vanilla RST processing"   )  
    parser.add_argument("--tags", default=d['tags'] )  
    parser.add_argument("--dev", action="store_true", default=d['dev'] )  
    parser.add_argument("-l","--level", default=d['level'], help="INFO/DEBUG/WARN/..")  
    
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.level.upper()))

    args.resolver = Resolver(args)

    return args





class DummyArgs(object):
    title = "DummyArgs"
    rstdir = "/tmp/env/trac2sphinx"
    tracdir = "/tmp/env/trac2sphinx"
    origtmpl = None
    origurl = None

    
if __name__ == '__main__':
    level = 'INFO'
    logging.basicConfig(level=getattr(logging, level)) 
    from env.sqlite.db import DB

    dbpath = sys.argv[1]
    pgname = sys.argv[2] 

    db = DB(dbpath)
    print db 
 
    args = DummyArgs()

    sph = Sphinx(args, db)
    sph.trac2rst_one(pgname)
    sph.write_(sph.pages[0])


