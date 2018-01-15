#!/usr/bin/env python
"""
tracwiki2rst.py
=================


"""
import logging, sys, re, os
log = logging.getLogger(__name__)
from env.sqlite.db import DB

class WikiPage(object):
    def __init__(self, db, name, encoding="UTF-8"):
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

        #self.author = author.encode(encoding)
        #self.text = text.encode(encoding)
        #self.comment = comment.encode(encoding)
        #assert type(self.author) is str
        #assert type(self.text) is str
        #assert type(self.comment) is str

        
    def __repr__(self):
        return "%5s : %30s : %10s : %15s : %60s : %s " % ( self.version, self.name, self.author, self.time, ",".join(self.tags), self.comment )

    def __str__(self):
        return "\n\n".join( [repr(self), repr(self.text) ] ) 


class Lines(list):
    def __init__(self, *args, **kwa):
        list.__init__(self, *args, **kwa)
    def __repr__(self):
        return "<%s %s lines> " % (self.__class__.__name__, len(self))
    def __str__(self):
        return "\n".join([repr(self)] + list(self))
    def indent(self, n):
        return map(lambda _:" "*n + _, list(self)) 
  
class Para(Lines):
    def as_rst(self):
        return "\n".join(self)

class Literal(Lines):
    start = "{{{"
    end = "}}}"

    @classmethod 
    def is_start(cls, line):
        return line.startswith(cls.start)
    @classmethod 
    def is_end(cls, line):
        return line.startswith(cls.end)

    def as_rst(self):
        return "\n".join(["","::", ""] + self.indent(4) )


class Toc(Lines):     
    def as_rst(self):
        return "\n".join(["",".. toctree::", ""] + self.indent(3) + ["",""] )


class Head(Lines):
    """
    http://docutils.sourceforge.net/docs/user/rst/quickref.html#section-structure
    """
    ptn = re.compile("^(=*) ([^=]*) (=*)$")
    mkr = list("=-~+#<>")   

    def as_rst(self):
        level = int(self.level)-1
        if level >= len(self.mkr):
            log.warning("Head level %d too large for %r  " % (level, self) ) 
            level = len(self.mkr)-1
        pass
        return "\n".join(["", self.title, self.mkr[level] * len(self.title) ])

    def __repr__(self):
        return "<Head %s %s %s lines> " % (self.title, self.level, len(self))

    @classmethod 
    def is_head(cls, line):
        m = cls.ptn.match(line)
        return m is not None
        #return len(line) > 0 and line[0] == "=" and line[-1] == "=" 

    @classmethod 
    def match(cls, line):
        m = cls.ptn.match(line)
        assert m , "match failed for line [%s] " % line
        a, title, b = m.groups()

        if len(a) != len(b):
           log.warning("got unequal a, b  %s %s %s " % (a,b, line))
        pass
        #assert a == b, (a, b, "expect same", line) 
        level = max(len(a), len(b))

        if int(level)-1 > len(cls.mkr):
            log.fatal("Head level %d too large for line [%s]  " % (level, line ))
            assert 0   
        pass
        return title, level

    @classmethod
    def from_line(cls, line):
        assert cls.is_head(line)
        title, level = cls.match(line) 
        head = cls(title, level)
        head.append(line)
        return head

    def __init__(self, title, level):
        self.title = title
        self.level = level



class Content(list):
    def __repr__(self):
        return "\n".join(map(repr, list(self)))

    def __str__(self):
        return "\n".join(map(str, list(self)))

    def as_rst(self):
        return "\n".join(map(lambda _:_.as_rst(), list(self)))


class Wiki2RST(object):
    """
    Other things to translate::

       [[ListTagged(Arc or Noah)]]

    """ 
    skips = r"""
    [[TracNav
    """.lstrip().rstrip().split()

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

class Page(Content):
    @classmethod
    def from_tracwiki(cls, wikipage, args, dbg=True):
        name = wikipage.name
        page = cls(name)

        if dbg:
            top = Para()
            top.append(":orphan:")
            top.append("")
            if args.origtmpl is not None:
                top.append("* %s " % (args.origtmpl % name) )
            pass
            page.append(top)
        pass

        conv = Wiki2RST(wikipage.text, page)

        if dbg:
            page.append(Head("original tracwiki text",1))
            page.append(Literal(wikipage.text.split("\r\n")))

            # BELOW ARE PROBLEMATIC FOR ENCODING
            #page.append(Head("Page content repr",1))
            #page.append(Literal(repr(page).split("\n")))

            #page.append(Head("Page content str",1))
            #page.append(Literal(str(page).split("\n")))
 
            #rst = page.as_rst()   # dont recurse 
            #page.append(Head("converted rst",1))
            #page.append(Literal(rst.split("\n")))
        pass
        return page 

    def __init__(self, name):
        Content.__init__(self)
        self.name = name




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
        rst = page.as_rst()
        open(rstpath, "w").write(rst.encode('utf-8'))   

    def write(self):
        idx = self.make_index("index", self.title)
        self.write_(idx) 
        for page in self.pages:
            self.write_(page)
        pass

    def make_index(self, name, title):
        idx = Page(name)        
        hdr = Head(title, 1)
        toc = Toc()
        for page in self.pages:
            toc.append(page.name)
        pass

        foot = Head("indices and tables", 1)

        para = Para()
        para.append("")
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
                wikipage = WikiPage(self.db, name)
                print repr(wikipage)
                txtpath = self.getpath(name, ".txt") 
                open(txtpath, "w").write(wikipage.text.encode('utf-8'))   

                page = Page.from_tracwiki(wikipage, self.args)
                self.add(page)
            pass
        else:
            wikipage = WikiPage(self.db, name)
            page = Page.from_tracwiki(wikipage, self.args) 
            self.add(page)
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
    sphinx = Sphinx(args, db)
    sphinx.trac2rst()
    sphinx.write()




