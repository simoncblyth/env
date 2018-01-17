#!/usr/bin/env python
"""
"""
import logging, sys, re, os, collections, datetime, codecs
log = logging.getLogger(__name__)
from env.sqlite.db import DB

from env.trac.migration.doclite import Para, Head, HorizontalRule, ListTagged, Toc, Literal, CodeBlock, Meta, Anchor, Contents, Sidebar, Page
from env.trac.migration.rsturl import EscapeURL

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



        
class Wiki2RST(object):
    """
    Other things to translate::

       [[ListTagged(Arc or Noah)]]

    """ 
    skips = r"""
    [[TracNav
    """.lstrip().rstrip().split()


    @classmethod
    def meta_top(cls, wp, pg, args):
        origurl = args.origtmpl % wp.name if args.origtmpl is not None else None

        md = wp.metadict

        anchor = Anchor(wp.name, md["tags"])
        pg.append(anchor)

        sidebar = Sidebar(md)
        pg.append(sidebar) 

        contents = Contents(depth=2)
        pg.append(contents)

        meta = Meta(md)
        meta.append(":orphan:")
        if not origurl is None:
            meta.append(":origurl: %s" % origurl)
        pass
        pg.append(meta)

        if not origurl is None:
            para = Para(["", "* %s " % origurl, ""])
            pg.append(para)
        pass

     
    @classmethod
    def dbg_tail(cls, wp, pg):
        """
        This would be wrong (fails with non-ascii) as it mixes byte strings and unicode::

           pg.append(Literal(str(pg).split("\n"))) 

        """
        rst = pg.rst   # do here to avoid including all the below debug additions to page

        pg.append(Head("%s dbg_tail" % wp.name,1))

        pg.append(Head("Literal converted rst",2))
        pg.append(CodeBlock(rst.split("\n"), lang="rst", linenos=True))

        pg.append(Head("Literal tracwiki text",2))
        pg.append(CodeBlock(wp.text.split("\n"),lang="bash", linenos=True))

        pg.append(Head("Literal repr(pg)",2))
        pg.append(CodeBlock(repr(pg).split("\n"), lang="pycon", linenos=True))

        pg.append(Head("Literal unicode(pg)",2))
        pg.append(CodeBlock(unicode(pg).split("\n"), lang="pycon", linenos=True))

    @classmethod
    def page_from_tracwiki(cls, wp, text, args, dbg=True):
        name = wp.name
        pg = Page(name)

        cls.meta_top(wp, pg, args) 

        conv = cls(text, pg, name=name)

        for i in pg.incomplete_instances():
            wp.complete_ListTagged(i)
        pass

        if dbg:
            cls.dbg_tail(wp, pg) 
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

    def __init__(self, text, content, name=None):
        self.orig = text
        self.content = content
        self.name = name

        lines = filter(lambda _:not(_.startswith(tuple(self.skips))), text.split("\n"))

        self.cur_para = None
        self.cur_literal = None

        for line in lines:
            # cur_literal None avoids looking inside literal blocks for Heads OR ListTagged
            if self.cur_literal is None and Head.is_match(line):  
                self.end_para()
                head = Head.from_line(line, name=self.name)
                self.content.append(head) 
            elif self.cur_literal is None and HorizontalRule.is_match(line):
                self.end_para()
                hr = HorizontalRule()
                self.content.append(hr) 
            elif self.cur_literal is None and ListTagged.is_match(line):
                self.end_para()
                tgls = ListTagged.from_line(line)
                self.content.append(tgls) 
            elif Literal.is_start(line):
                self.end_para()
                self.cur_literal = Literal()
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
        log.debug("write %s " % rstpath )
        rst = page.rst
        assert type(rst) is unicode
        open(rstpath, "w").write(rst.encode("utf-8"))  

    def write(self):
        idx = self.make_index("index", self.title)
        for page in self.pages:
            self.write_(page)
        pass
        self.write_(idx) 
        log.info("wrote %s pages  %s...%s " % (len(self.pages), self.pages[0].name, self.pages[-1].name))

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

    def trac2rst_one(self, name):
        log.debug("converting %s " % name )
        txtpath = self.getpath(name, ".txt") 

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

        pg = Wiki2RST.page_from_tracwiki(wp, use_text, self.args)
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
    d['title'] = "tracwiki2rst.py conversion"
    d['origtmpl'] = None
    d['level'] = "INFO"
    d['dev'] = False
    d['tags'] = None
    #d['txtsrc'] = False

    parser.add_argument("dbpath", default=None, help="path to trac.db"  ) 
    parser.add_argument("--onepage", default=d['onepage'], help="restrict conversion to single named page for debugging")  
    parser.add_argument("--rstdir", default=d['rstdir'], help="directory to write the converted RST")  
    parser.add_argument("--origtmpl", default=d['origtmpl'], help="template of original tracwiki url to provide backlink for debugging, eg http://localhost/tracs/worklow/wiki/%s ")  
    parser.add_argument("--title", default=d['title'] )  
    #parser.add_argument("--txtsrc", action="store_true", default=d['txtsrc'], help="Instead of getting wiki text from the scm backup trac.db, get from .txt file"   )  
    parser.add_argument("--tags", default=d['tags'] )  
    parser.add_argument("--dev", action="store_true", default=d['dev'] )  
    parser.add_argument("-l","--level", default=d['level'], help="INFO/DEBUG/WARN/..")  
    
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.level.upper()))
    return args


if __name__ == '__main__':
    args = parse_args(__doc__)
    dbpath = args.dbpath
    db = DB(dbpath)
    print db 



