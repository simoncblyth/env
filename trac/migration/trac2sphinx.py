#!/usr/bin/env python
"""
trac2sphinx.py 
================

Canonical usage is from wtracdb.py 

::

    wtracdb- ; ./trac2sphinx.py $(wtracdb-path) --onepage 3D --level DEBUG
    wtracdb- ; ./trac2sphinx.py $(wtracdb-path) --onepage 3D --level debug --logformat "%(message).100s"   ## truncate message length 
    wtracdb- ; ./trac2sphinx.py $(wtracdb-path) --onepage 3D --level debug -F0
    wtracdb- ; ./trac2sphinx.py $(wtracdb-path) --onepage 3D --level debug -F1
    wtracdb- ; ./trac2sphinx.py $(wtracdb-path) --onepage 3D --level debug -F2  ## shorthand way to pick logformat 

    wtracdb- ; ./trac2sphinx.py $(wtracdb-path) --onepage WikiFormatting --level debug -F2
    wtracdb- ; ./trac2sphinx.py $(wtracdb-path) --onepage WikiFormatting -LD -F2


"""
import logging, sys, re, os, collections, datetime, codecs, copy
log = logging.getLogger(__name__)

from env.sqlite.db import DB
from env.trac.migration.xmlrpcproxy import Proxy

from env.trac.migration.resolver import Resolver
from env.trac.migration.tracwikipage import TracWikiPage
from env.trac.migration.tracwiki2rst import TracWiki2RST
from env.trac.migration.inlinetracwiki2rst import InlineTrac2Sphinx
from env.doc.extlinks import SphinxExtLinks


class Trac2Sphinx(object):
    @classmethod
    def make_context(cls, doc, extlinks={}):
        """
        The extlinks dict shold match that supplied to Sphinx in conf.py.
        """
        import argparse
        parser = argparse.ArgumentParser(doc)

        d = {}
        d['onepage'] = None
        d['rstdir'] = "/tmp/env/trac2sphinx"
        d['tracdir'] = "/tmp/env/trac2sphinx"
        d['title'] = u"trac2sphinx.py conversion"
        d['origtmpl'] = None
        d['dev'] = False
        d['tags'] = None
        d['vanilla'] = False

        fmt = {} 
        fmt['0'] = "%(asctime)-15s %(levelname)-7s %(name)-20s:%(lineno)-3d %(message)s"
        fmt['1'] = "%(levelname).1s %(name)-20s:%(lineno)-3d %(message).100s"
        fmt['2'] = "%(levelname).1s %(lineno)-3d %(message).200s"
        d['logformat'] = '2'

        lvl = dict(I="INFO",D="DEBUG",W="WARN")
        d['level'] = "I"
    
        parser.add_argument("dbpath", default=None, help="path to trac.db"  ) 
        parser.add_argument("--onepage", default=d['onepage'], help="restrict conversion to single named page for debugging")  

        # TODO: get these from an ini file, not commandline as rather constant 
        parser.add_argument("--rstdir", default=d['rstdir'], help="directory to write the converted RST")  
        parser.add_argument("--tracdir", default=d['tracdir'], help="directory named after the repo containing db/trac.db as well as attachements etc.. ")  
        parser.add_argument("--title", default=d['title'] )  

        # TODO: eliminate using the extlinks 
        parser.add_argument("--origtmpl", default=d['origtmpl'], help="template of original tracwiki url to provide backlink for debugging, eg http://localhost/tracs/worklow/wiki/%s ")  

        parser.add_argument("--vanilla", action="store_true", default=d['vanilla'], help="Skip Sphinx extensions to allow plain vanilla RST processing"   )  
        parser.add_argument("--tags", default=d['tags'] )  
        parser.add_argument("--dev", action="store_true", default=d['dev'] )  
        parser.add_argument("-F","--logformat", default=d['logformat'] )
        parser.add_argument("-L","--level", default=d['level'], help="I/D/W/INFO/DEBUG/WARN/..")  
        
        ctx = parser.parse_args()
        logging.basicConfig(format=fmt.get(ctx.logformat, ctx.logformat), level=getattr(logging, lvl.get(ctx.level,ctx.level).upper()))

        ctx.resolver = Resolver(tracdir=ctx.tracdir, rstdir=ctx.rstdir)
        ctx.db = DB(ctx.dbpath)
        ctx.proxy = Proxy.create("workflow_trac", "~/.env.cnf")
        ctx.extlinks = SphinxExtLinks(extlinks)
        ctx.inliner_ = InlineTrac2Sphinx(ctx)
        ctx.stats = collections.defaultdict(lambda:0)

        return ctx

    def __init__(self, ctx):
        self.ctx = ctx
        self.pages = []
        self.dnames = sorted(map(lambda _:_[0],self.ctx.db("select distinct name from wiki")))
        self.pnames = sorted(map(unicode,self.ctx.proxy.pages)) if self.ctx.proxy is not None else []
        if len(self.pnames) > 0:
            self.compare_names()
            self.names = self.pnames
            self.ctx.skipdoc = self.dbonly   ## web interface ahead of db, so these are deletes thru web interface
            log.warning("adopt wiki page list from xmlrpc proxy ")
        else:
            self.names = self.dnames
            self.ctx.skipdoc = []
        pass
            
    def compare_names(self):
        log.info("dnames[:10] " + str(self.dnames[:10]))
        log.info("pnames[:10] " + str(self.pnames[:10]))
        log.info("compare_db_proxy db %s pr %s " % (len(self.dnames), len(self.pnames)))
        self.dbonly = list(set(self.dnames) - set(self.pnames))  ## db (from scm backup) often behind live instance
        self.pronly = list(set(self.pnames) - set(self.dnames))
        log.info("db only  %s : %s " % (len(self.dbonly), str(self.dbonly)) )
        log.info("pr only  %s : %s " % (len(self.pronly), str(self.pronly)) )
        pass

    def add(self, page):
        self.pages.append(page)  
 
    def write_text(self, text, path):
        log.debug("write_text %s " % path)
        assert type(text) is unicode
        open(path, "w").write(text.encode("utf-8"))  

    def read_text(self, path):
        log.debug("read_text %s " % path)
        text = codecs.open(path, encoding='utf-8').read() if os.path.exists(path) else None
        assert type(text) in [unicode, type(None)], (txtpath, type(text_from_file) )
        return text

    def write_(self, page):
        path = self.ctx.resolver.getpath(page.name, ".rst") 
        self.write_text(page.rst, path )
 
    def write(self):
        for page in self.pages:
            self.write_(page)
        pass
        pagenames = map(lambda _:_.name,self.pages)
        idx = TracWiki2RST.make_index("index", self.ctx.title, pagenames, ctx=self.ctx)
        self.write_(idx) 
        log.info("wrote %s pages  %s...%s " % (len(self.pages), self.pages[0].name, self.pages[-1].name))

    def trac2rst_one(self, name):
        """
        If a text file of the relevant path exists it overrides the 
        text grabbed from the DB, other things like the tags are still sourced
        from the DB.

        This was done to facilitate correcting issues with the source trac wiki text
        using the Trac instance web interface and then updating the .txt over xmlrpc.

        Once all such edits are done, can make another scm backup and get all text from
        the backup trac.db
        """
        log.debug("converting %s " % name )
        txtpath = self.ctx.resolver.getpath(name, ".txt") 

        wp = TracWikiPage(self.ctx.db, name)
        text_from_db = wp.text
        assert type(text_from_db) is unicode

        text_from_file = self.read_text(txtpath)

        if text_from_file:
            if text_from_db != text_from_file:
                log.debug("difference between wikitext from db and xmlrpc for %s " % name)
            pass
        pass
        use_text = text_from_file if text_from_file is not None else text_from_db 

        pg = TracWiki2RST.page_from_tracwiki(wp, use_text, self.ctx)
        self.add(pg)
       

    def trac2rst_all(self):
        for name in self.names:
            self.trac2rst_one(name)
        pass
        self.dumpstats()


    def dumpstats(self):
        log.info("dumpstats")
        for k,v in self.ctx.stats.items():
            print " %20s : %s " % (k, v )
        pass

    def tracdb2rst(self):
        name = self.ctx.onepage
        if name is None:
            self.trac2rst_all()
        else:
            self.trac2rst_one(name)
        pass


    
if __name__ == '__main__':


    extlinks = {}
    ctx = Trac2Sphinx.make_context(__doc__, extlinks)

    t2s = Trac2Sphinx(ctx)
    t2s.trac2rst_one(t2s.ctx.onepage)
    t2s.write_(t2s.pages[0])


