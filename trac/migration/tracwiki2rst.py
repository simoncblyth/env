#!/usr/bin/env python
"""
tracwiki2rst.py
=================

::

   ./tracwiki2rst.py $(wtracdb-path) --onepage 3D


"""

import logging, sys, re, os
log = logging.getLogger(__name__)
from env.sqlite.db import DB

from env.trac.migration.doclite import Para, Head, Toc, Literal, Page


class WikiPage(object):
    def __init__(self, db, name):
        self.name = name
        self.tags = map(lambda _:str(_[0]), db("select tag from tags where tagspace=\"wiki\" and name=\"%s\" " % name ))

        rec = db("SELECT version,time,author,text,comment,readonly FROM wiki WHERE name=\"%s\" ORDER BY version DESC LIMIT 1" % name ) 
        version,time,author,text,comment,readonly = rec[0] 

        self.version = version
        self.time = time

        assert type(author) is unicode
        assert type(text) is unicode
        if comment is not None:
            assert type(comment) is unicode
        pass

        self.author = author
        self.text = text
        self.comment = comment

              
    def __repr__(self):
        return "%5s : %30s : %10s : %15s : %60s : %s " % ( self.version, self.name, self.author, self.time, ",".join(self.tags), self.comment )

    def __unicode__(self):
        return "\n\n".join( [repr(self), unicode(self.text) ] ) 

    def __str__(self):
        return unicode(self).encode('utf-8')


class Wiki2RST(object):
    """
    Other things to translate::

       [[ListTagged(Arc or Noah)]]

    """ 
    skips = r"""
    [[TracNav
    """.lstrip().rstrip().split()


    @classmethod
    def page_from_tracwiki(cls, wp, args, dbg=True):
        name = wp.name
        pg = Page(name)

        if dbg:
            top = Para()
            top.append(":orphan:")
            top.append("")
            if args.origtmpl is not None:
                top.append("* %s " % (args.origtmpl % name) )
            pass
            pg.append(top)
        pass

        conv = cls(wp.text, pg)

        if dbg:
            pg.append(Head("original tracwiki text",1))
            pg.append(Literal(wp.text.split("\r\n")))

            # BELOW REQUIRED CLEAR THINKING REGARDS ENCODINGS
            pg.append(Head("Page content repr",1))
            pg.append(Literal(repr(pg).split("\n")))

            pg.append(Head("Page content str",1))
            #pg.append(Literal(str(pg).split("\n")))  # << mixing byte strings and unicode is unhealthy 
            pg.append(Literal(unicode(pg).split("\n")))
 
            rst = pg.rst   # dont recurse 
            pg.append(Head("converted rst",1))
            pg.append(Literal(rst.split("\n")))
        pass
        return pg 

    def end_para(self):
        if self.cur_para is None:return
        self.content.append(self.cur_para)
        self.cur_para = None
    def end_literal(self):
        if self.cur_literal is None:return
        self.content.append(self.cur_literal)
        self.cur_literal = None

    def __init__(self, text, content):
        self.orig = text
        self.content = content

        log.debug("Wiki2RST __init__ " ) 

        lines = filter(lambda _:not(_.startswith(tuple(self.skips))), text.split("\r\n"))

        self.cur_para = None
        self.cur_literal = None

        for line in lines:
            if self.cur_literal is None and Head.is_head(line):  # avoid looking for head inside literal blocks 
                head = Head.from_line(line)
                self.content.append(head) 
                self.end_para()
            elif Literal.is_start(line):
                self.cur_literal = Literal()
                self.end_para()
            elif Literal.is_end(line):
                self.end_literal()
            else:
                if self.cur_literal is not None:
                    self.cur_literal.append(line)
                else:
                    if self.cur_para is None:
                         self.cur_para = Para()
                    pass
                    self.cur_para.append(line) 
                pass 
            pass
        pass
        self.end_para()




class Sphinx(object):
    def __init__(self, args, db):
        self.args = args 
        self.db = db
        self.base = args.rstdir
        log.info("rstdir:%s" % self.base)
        self.title = args.title
        self.pages = []

    def getpath(self, name, ext=".rst"):
        path = os.path.join(self.base, "%s%s" % (name,ext) )
        dir_ = os.path.dirname(path)
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        pass
        return path

    def add(self, page):
        self.pages.append(page)  

    def write_(self, page):
        """
        http://www.sphinx-doc.org/en/stable/rest.html#source-encoding
        Sphinx assumes source files to be encoded in UTF-8 by default
        """
        rstpath = self.getpath(page.name, ".rst") 
        log.info("write %s " % rstpath )
   
        rst = page.rst
        assert type(rst) is unicode
        open(rstpath, "w").write(rst.encode("utf-8"))  

    def write(self):
        idx = self.make_index("index", self.title)
        for page in self.pages:
            self.write_(page)
        pass
        self.write_(idx) 

    def make_index(self, name, title):
        idx = Page(name)        
        hdr = Head(title, 1)
        toc = Toc()
        for page in self.pages:
            toc.append(page.name)
        pass

        foot = Head("indices and tables", 1)

        para = Para()
        para.append(u"")   # some unicode is needed, to get the py2 coercion to kick in 
        para.append("* :ref:`genindex`")
        para.append("* :ref:`modindex`")
        para.append("* :ref:`modindex`")
        para.append("")

        idx.append(hdr) 
        idx.append(toc) 
        idx.append(foot) 
        idx.append(para) 

        return idx

    def trac2rst(self):
        name = self.args.onepage
        if name is None:
            names = self.db("select distinct name from wiki") 
            for name, in names:
                log.debug("converting %s " % name )
                wp = WikiPage(self.db, name)
                print repr(wp)
                txtpath = self.getpath(name, ".txt") 
                assert type(wp.text) is unicode
                open(txtpath, "w").write(wp.text.encode('utf-8'))   

                pg = Wiki2RST.page_from_tracwiki(wp, self.args)
                self.add(pg)
            pass
        else:
            wp = WikiPage(self.db, name)
            pg = Wiki2RST.page_from_tracwiki(wp, self.args) 
            self.add(pg)
        pass



def parse_args(doc):
    import argparse
    parser = argparse.ArgumentParser(doc)

    d = {}
    d['onepage'] = None
    d['rstdir'] = None
    d['title'] = "tracwiki2rst.py conversion"
    d['origtmpl'] = None
    d['level'] = "INFO"
    d['dev'] = False

    parser.add_argument("dbpath", default=None, help="path to trac.db"  ) 
    parser.add_argument("--onepage", default=d['onepage'], help="restrict conversion to single named page for debugging")  
    parser.add_argument("--rstdir", default=d['rstdir'], help="directory to write the converted RST")  
    parser.add_argument("--origtmpl", default=d['origtmpl'], help="template of original tracwiki url to provide backlink for debugging, eg http://localhost/tracs/worklow/wiki/%s ")  
    parser.add_argument("--title", default=d['title'] )  
    parser.add_argument("--dev", action="store_true", default=d['dev'] )  
    parser.add_argument("-l","--level", default=d['level'], help="INFO/DEBUG/WARN/..")  
    
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.level.upper()))
    return args


if __name__ == '__main__':
    args = parse_args(__doc__)
    dbpath = args.dbpath
    db = DB(dbpath)

    if args.onepage is not None:
        wp = WikiPage(db, args.onepage)
        print str(wp)
        pg = Wiki2RST.page_from_tracwiki(wp, args)
        print str(pg)
    else:
        sphinx = Sphinx(args, db)
        sphinx.trac2rst()
        sphinx.write()
    pass




